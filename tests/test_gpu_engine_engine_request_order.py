"""Literal metadata/party arrays only; never initialize or step an engine."""
import copy
import itertools
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_request_order as r
from gpu_engine.engine_private_team import OwnedTeam


class RequestOrderTests(unittest.TestCase):
    def order(self, sizes=(6, 4)):
        return r.initialize_order(j.array([7, 7], j.int32), j.array([0, 1], j.int32),
            j.array(sizes, j.int32), engine_binding=j.zeros((2, 8), j.uint32))

    def receipt(self, order, incoming, *, installed=(True, True)):
        return r.SwitchInstalled(order.stream_id, order.owner_side, order.engine_binding,
            order.generation, j.array(installed), order.request_to_physical[:, 0], j.array(incoming, j.int32))

    def boundary(self, order):
        return r.RequestBoundary(order.stream_id, order.owner_side, order.engine_binding,
            order.generation, order.request_to_physical[:, 0])

    def route(self, order, choices, **changes):
        batch = order.owner_side.shape[0]
        args = dict(boundary=self.boundary(order), requested=j.ones(batch, j.bool_),
            request_legal=j.ones((batch, 9), j.bool_), engine_legal=j.ones((batch, 9), j.bool_),
            move_action_map=j.broadcast_to(j.arange(4, dtype=j.int32), (batch, 4)))
        args.update(changes)
        return r.route_choice(order, j.array(choices, j.int32), **args)

    def owned(self, order):
        batch = order.owner_side.shape[0]
        marker = j.broadcast_to(j.arange(1, 7, dtype=j.int32), (batch, 6))
        values = []
        for index, field in enumerate(OwnedTeam._fields[2:]):
            if field == 'active':
                value = order.request_to_physical[:, 0]
            elif field == 'stats':
                value = marker[..., None] * 100 + j.arange(6, dtype=j.int32)
            elif field == 'species':
                # Deliberate duplicate species prove routing never joins on ID.
                value = j.where(j.arange(6) < order.team_size[:, None], j.where(marker == 3, 1, marker), 0)
            else:
                value = marker + 1000 * (index + 1)
            values.append(value)
        return OwnedTeam(order.owner_side, order.engine_binding, *values)

    def test_source_swap_order_tracks_installed_switches_and_ignores_cancelled_choices(self):
        order = self.order()
        source = [list(range(6)), list(range(4))]
        for incoming, installed in (((3, 2), (True, True)), ((5, -999), (True, False)),
                ((1, 0), (True, True)), ((0, 3), (True, True))):
            receipt = self.receipt(order, incoming, installed=installed)
            previous = order
            order = r.install_switch(order, receipt)
            for row in range(2):
                if installed[row]:
                    position = source[row].index(incoming[row])
                    source[row][0], source[row][position] = source[row][position], source[row][0]
                self.assertEqual(order.request_to_physical[row].tolist(), source[row] + [-1] * (6-len(source[row])))
            np.testing.assert_array_equal(order.generation, previous.generation + np.array(installed, np.int32))
            self.assertEqual(order.admitted.tolist(), [True, True])

    def test_all_six_member_request_permutations_route_to_engine_bench_order(self):
        permutations = list(itertools.permutations(range(6)))
        party = [permutation for permutation in permutations for _ in range(5)]
        choices = [choice for _ in permutations for choice in range(4, 9)]
        batch = len(party)
        order = r.initialize_order(j.arange(1, batch+1, dtype=j.int32), j.zeros(batch, j.int32),
            j.full(batch, 6, j.int32), engine_binding=j.zeros((batch, 8), j.uint32))
        order = order._replace(request_to_physical=j.array(party, j.int32))
        got = self.route(order, choices)
        expected_targets = [p[c-3] for p, c in zip(party, choices)]
        expected_actions = [4 + [slot for slot in range(6) if slot != p[0]].index(target)
            for p, target in zip(party, expected_targets)]
        self.assertTrue(bool(j.all(got.admitted)))
        self.assertEqual(got.physical_switch_target.tolist(), expected_targets)
        self.assertEqual(got.engine_action.tolist(), expected_actions)

    def test_switch_sequence_replay_wrong_identity_and_bad_party_poison_only_affected_owner(self):
        initial = self.order()
        receipt = self.receipt(initial, (3, 2))
        advanced = r.install_switch(initial, receipt)
        repeated = r.install_switch(advanced, receipt)
        self.assertEqual(repeated.admitted.tolist(), [False, False])
        self.assertFalse(bool(j.any(self.route(repeated, [4, 4]).admitted)))
        cases = [receipt._replace(stream_id=receipt.stream_id.at[0].add(1)),
            receipt._replace(owner_side=receipt.owner_side.at[0].set(1)),
            receipt._replace(engine_binding=receipt.engine_binding.at[0, 0].add(j.uint32(1))),
            receipt._replace(incoming_physical=receipt.incoming_physical.at[0].set(6)),
            receipt._replace(incoming_physical=receipt.incoming_physical.at[0].set(0)),
            receipt._replace(outgoing_physical=receipt.outgoing_physical.at[0].set(2))]
        for bad in cases:
            got = r.install_switch(initial, bad)
            self.assertEqual(got.admitted.tolist(), [False, True])
            self.assertEqual(got.request_to_physical[0].tolist(), [-1]*6)
            healed = r.install_switch(got, self.receipt(got, (1, 1)))
            self.assertFalse(bool(healed.admitted[0]))

    def test_move_routing_is_explicit_and_unknown_maps_waits_or_masks_never_dispatch(self):
        order = self.order()
        mapping = j.array([[3, -1, -1, -1], [2, -1, -1, -1]], j.int32)
        got = self.route(order, [0, 0], move_action_map=mapping)
        self.assertEqual(got.engine_action.tolist(), [3, 2])
        self.assertEqual(got.physical_switch_target.tolist(), [-1, -1])
        for changes in (dict(move_action_map=j.full((2, 4), -1, j.int32)),
                dict(requested=j.array([False, False])),
                dict(request_legal=j.zeros((2, 9), j.bool_)),
                dict(engine_legal=j.zeros((2, 9), j.bool_))):
            got = self.route(order, [0, 0], **changes)
            self.assertEqual(got.engine_action.tolist(), [-1, -1])
            self.assertEqual(got.admitted.tolist(), [False, False])
        got = self.route(order, [-1, 9])
        self.assertEqual(got.admitted.tolist(), [False, False])
        got = self.route(order, [8, 8])
        self.assertEqual(got.admitted.tolist(), [True, False])

    def test_reordering_every_owner_field_uses_private_indices_and_preserves_padding(self):
        order = self.order()
        order = r.install_switch(order, self.receipt(order, (3, 2)))
        owned = self.owned(order)
        got = r.reorder_owned_team(owned, order, boundary=self.boundary(order))
        self.assertEqual(got.admitted.tolist(), [True, True])
        for row in range(2):
            indices = [x for x in order.request_to_physical[row].tolist() if x >= 0]
            for field in OwnedTeam._fields[2:]:
                if field == 'active':
                    self.assertEqual(int(got.owned.active[row]), 0)
                    continue
                expected = np.zeros_like(np.asarray(getattr(owned, field)[row]))
                expected[:len(indices)] = np.asarray(getattr(owned, field)[row])[indices]
                np.testing.assert_array_equal(getattr(got.owned, field)[row], expected, err_msg=field)

    def test_wrong_request_identity_active_or_permutation_emits_no_private_facts(self):
        order = self.order()
        owned = self.owned(order)
        boundary = self.boundary(order)
        for bad_boundary in (boundary._replace(stream_id=boundary.stream_id.at[0].add(1)),
                boundary._replace(owner_side=boundary.owner_side.at[0].set(1)),
                boundary._replace(generation=boundary.generation.at[0].add(1)),
                boundary._replace(active_physical=boundary.active_physical.at[0].set(3)),
                boundary._replace(engine_binding=boundary.engine_binding.at[0, 0].add(j.uint32(1)))):
            got = r.reorder_owned_team(owned, order, boundary=bad_boundary)
            self.assertEqual(got.admitted.tolist(), [False, True])
            self.assertEqual(int(got.owned.owner_side[0]), -1)
            for field in OwnedTeam._fields[2:]:
                self.assertFalse(bool(j.any(getattr(got.owned, field)[0])), field)
            self.assertEqual(self.route(order, [4, 4], boundary=bad_boundary).admitted.tolist(), [False, True])
        for bad in (owned._replace(active=owned.active.at[0].set(1)),
                owned._replace(owner_side=owned.owner_side.at[0].set(1))):
            got = r.reorder_owned_team(bad, order, boundary=boundary)
            self.assertEqual(got.admitted.tolist(), [False, True])
        bad_order = order._replace(request_to_physical=order.request_to_physical.at[0, 1].set(0))
        self.assertFalse(bool(self.route(bad_order, [4, 4]).admitted[0]))

    def test_reordered_owner_facts_match_source_parser_after_different_switches(self):
        from test_gpu_engine_engine_private_team import EnginePrivateTeamTests
        from gpu_engine import engine_private_team as e, private_team as p
        EnginePrivateTeamTests.setUpClass()
        fixture = EnginePrivateTeamTests('test_every_team_fact_and_owned_writer_column_matches_source_both_owners')
        exposure = e.RequestExposure('base', 'physical', True, True)
        compiled = fixture.tables(exposure)
        binding = j.broadcast_to(e.engine_metadata_binding(fixture.metadata), (2, 8))
        order = r.initialize_order(j.array([9, 9], j.int32), j.array([0, 1], j.int32),
            j.array([3, 3], j.int32), engine_binding=binding)
        order = r.install_switch(order, self.receipt(order, (2, 1)))
        physical = fixture.owned()._replace(active=j.array([2, 1], j.int32))
        gathered = r.reorder_owned_team(physical, order, boundary=self.boundary(order))
        got = e.team_facts(gathered.owned, compiled, expected_binding=compiled.binding)
        packets = fixture.requests(exposure)
        for row, packet in enumerate(packets):
            original = packet['side']['pokemon']
            packet['side']['pokemon'] = [copy.deepcopy(original[i]) for i in order.request_to_physical[row].tolist() if i >= 0]
            for index, mon in enumerate(packet['side']['pokemon']):mon['active'] = index == 0
        expected = jax.tree.map(lambda *x:j.stack(x), *[p.parse_team_facts(packet,
            owner_side=side, vocabulary=fixture.vocab, dex=fixture.dex) for side, packet in enumerate(packets)])
        self.assertEqual(got.admitted.tolist(), [True, True])
        for name in expected._fields:
            np.testing.assert_array_equal(getattr(got.facts, name), getattr(expected, name), err_msg=name)

    def test_literal_jit_outputs_match_eager_and_counter_overflow_refuses(self):
        order = self.order()
        receipt = self.receipt(order, (3, 2))
        eager = r.install_switch(order, receipt)
        fast = jax.jit(r.install_switch)(order, receipt)
        for a, b in zip(eager, fast):np.testing.assert_array_equal(a, b)
        owned = self.owned(fast)
        boundary = self.boundary(fast)
        eager_team = r.reorder_owned_team(owned, fast, boundary=boundary)
        fast_team = jax.jit(r.reorder_owned_team)(owned, fast, boundary=boundary)
        for a, b in zip(jax.tree.leaves(eager_team), jax.tree.leaves(fast_team)):np.testing.assert_array_equal(a, b)
        options = dict(boundary=boundary, requested=j.ones(2, j.bool_),
            request_legal=j.ones((2, 9), j.bool_), engine_legal=j.ones((2, 9), j.bool_),
            move_action_map=j.broadcast_to(j.arange(4, dtype=j.int32), (2, 4)))
        eager_choice = r.route_choice(fast, j.array([4, 6], j.int32), **options)
        fast_choice = jax.jit(r.route_choice)(fast, j.array([4, 6], j.int32), **options)
        for a, b in zip(eager_choice, fast_choice):np.testing.assert_array_equal(a, b)
        saturated = order._replace(generation=j.full(2, 2**31-1, j.int32))
        self.assertEqual(r.install_switch(saturated, self.receipt(saturated, (3, 2))).admitted.tolist(), [False, False])

    def test_registration_and_device_types_never_coerce_invalid_owner_or_choice(self):
        order = self.order()
        with self.assertRaises(ValueError):
            r.initialize_order(j.array(1, j.int32), j.array(0, j.int32), j.array(6, j.int32), engine_binding=j.zeros((1, 8), j.uint32))
        for owners in (j.array([True, False]), j.array([0., 1.]), j.array([0, 2], j.int32), j.array([0, 0], j.int32)):
            with self.assertRaises(ValueError):
                r.initialize_order(order.stream_id, owners, order.team_size, engine_binding=order.engine_binding)
        with self.assertRaises(ValueError):
            r.route_choice(order, j.array([0., 1.]), boundary=self.boundary(order), requested=j.ones(2, j.bool_),
                request_legal=j.ones((2, 9), j.bool_), engine_legal=j.ones((2, 9), j.bool_), move_action_map=j.zeros((2, 4), j.int32))


if __name__ == '__main__':unittest.main()
