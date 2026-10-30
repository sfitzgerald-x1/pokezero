"""Tiny authored device/host/source component parity, never games or timings."""
from dataclasses import replace
import unittest
import numpy as np

import test_gpu_engine_engine_public_events as fixture_module


class EnginePublicEventDeviceTests(fixture_module.EnginePublicEventsTests):
    # Reuse fixture helpers, not the parent test class's discovered tests.
    @classmethod
    def setUpClass(cls):
        fixture_module.EnginePublicEventsTests.setUpClass.__func__(cls)
        from gpu_engine import engine_public_event_device as d
        cls.d = d
        cls.device_projector = d.compile_projector(cls.producer, expected_binding=cls.producer.binding,
            expected_engine_binding=np.array(cls.producer.engine_binding, np.uint32))

    def device_registry(self, batch=1):
        j = self.j
        return self.d.register_registry(stream_id=j.arange(1, batch+1, dtype=j.int32),
            engine_binding=j.broadcast_to(j.array(self.producer.engine_binding, j.uint32), (batch, 8)),
            generation=j.zeros((batch, 2), j.int32), physical_active=j.tile(j.array([[3, 5]], j.int32), (batch, 1)),
            public_slot=j.tile(j.array([[2, 4]], j.int32), (batch, 1)),
            encounter=j.tile(j.array([[17, 29]], j.int32), (batch, 1)),
            revealed_species_id=j.tile(j.array([[11, 22]], j.int32), (batch, 1)),
            established=j.ones((batch, 2), bool))

    def observer(self, registry, owners=None):
        batch = registry.stream_id.shape[0]
        return self.d.ObserverContext(registry.stream_id,
            self.j.zeros(batch, self.j.int32) if owners is None else owners,
            self.j.full((batch,), 3, self.j.int32), registry.generation)

    def device(self, tape, registry=None, observer=None, tables=None):
        registry = self.device_registry(tape.words.shape[0]) if registry is None else registry
        observer = self.observer(registry) if observer is None else observer
        return self.d.project_device(tape, self.device_projector.device if tables is None else tables,
            registry, observer)

    def parity(self, tape, registry, observer, result, catalog=None):
        from gpu_engine.public_inputs import pack_chunks
        from gpu_engine.public_context_protocol import PublicAddress
        catalog = self.producer if catalog is None else catalog
        streams = []
        for row in range(tape.words.shape[0]):
            active = {side: (int(registry.physical_active[row, side]), PublicAddress(side,
                int(registry.public_slot[row, side]), int(registry.encounter[row, side])))
                for side in (0, 1) if registry.established[row, side]}
            host = self.e.project(tape, battle=row, stream_id=int(observer.stream_id[row]),
                owner=int(observer.owner[row]), next_sequence=int(observer.next_sequence[row]), catalog=catalog,
                registry=self.e.PublicRegistry(active), expected_binding=catalog.binding,
                expected_engine_binding=np.array(catalog.engine_binding, np.uint32))
            streams.append(host.events)
            self.assertEqual(int(result.refusal[row]), host.refusal)
        expected = next(pack_chunks(np.array(observer.stream_id), streams,
            source_binding=np.array(catalog.source_binding, np.uint32), move_binding=np.array(catalog.move_binding, np.uint32)))
        for field in expected._fields:
            old, new = getattr(expected, field), getattr(result.chunk, field)
            if old is None:
                self.assertIsNone(new, field)
            else:
                np.testing.assert_array_equal(old, new, err_msg=field)
        self.assertTrue(bool(result.component_admitted.all()))
        self.assertFalse(bool(result.full_line_consumers_complete.any()))
        self.assertFalse(bool(result.full_encoder_admitted.any()))
        self.assertFalse(bool(result.B4_qualified.any()))
        self.assertEqual(result.error.tolist(), [0]*tape.words.shape[0])
        np.testing.assert_array_equal(result.next_sequence, observer.next_sequence+tape.count)

    def test_actual_callback_mixed_batch_both_owners_hot_jit_exact_host_abi(self):
        state = self.state(4)
        state = state._replace(hp=state.hp.at[0, 1, 5].set(1).at[1, 1, 5].set(1),
            status=state.status.at[0, 1, 5].set(4).at[1, 1, 5].set(4))
        actions = self.j.array([[0, 0], [0, 0], [9, 0], [0, 0]], self.j.int32)
        traced = self.trace(state, actions, chance_inputs=self.chance(4))
        registry = self.device_registry(4)
        observer = self.observer(registry, self.j.array([0, 1, 0, 1], self.j.int32))
        compiled = self.jax.jit(self.d.project_device)
        result = compiled(traced.callbacks, self.device_projector.device, registry, observer)
        self.parity(traced.callbacks, registry, observer, result)
        self.assertEqual(result.chunk.words.shape, (4, 64, 16))
        self.assertEqual(result.chunk.valid.sum(axis=1).tolist(), [3, 3, 0, 4])
        self.assertEqual(result.chunk.words[0, :3, 0].tolist(), [15, 10, 12])
        self.assertEqual(result.chunk.words[0, 0, 4:8].tolist(), [0, 2, 1, 4])
        fresh = self.trace(self.state(4), self.j.zeros((4, 2), self.j.int32), chance_inputs=self.chance(4))
        hot = compiled(fresh.callbacks, self.device_projector.device, registry, observer)
        self.parity(fresh.callbacks, registry, observer, hot)
        self.assertEqual(compiled._cache_size(), 1)

    def test_actual_substitute_damage_does_not_emit_private_hp(self):
        state = self.state()._replace(substitute_hp=self.state().substitute_hp.at[0, 1, 5].set(100))
        tape = self.trace(state, self.j.zeros((1, 2), self.j.int32), chance_inputs=self.chance()).callbacks
        registry = self.device_registry()
        observer = self.observer(registry)
        result = self.device(tape, registry, observer)
        self.parity(tape, registry, observer, result)
        self.assertEqual(result.chunk.words[0, :3, 0].tolist(), [15, 15, 10])
        self.assertEqual(result.chunk.words[0, 2, 6].item(), 0)

    def test_hp_source_rounding_status_and_zero_fnt_both_owners(self):
        batch = 8
        j = self.j
        tape = self.e.initialize(batch, capacity=1, catalog=self.producer)
        hp = j.array([999, 999, 1, 1, 1000, 1000, 0, 0], j.int32)
        status = j.array([4, 4, 5, 5, 0, 0, 4, 4], j.int32)
        tape = self.e.condition(tape, j.ones(batch, j.int32), j.full((batch,), 5, j.int32), hp,
            j.full((batch,), 1000, j.int32), status, eligible=j.ones(batch, bool))
        registry = self.device_registry(batch)
        observer = self.observer(registry, j.array([0, 1]*4, j.int32))
        result = self.jax.jit(self.d.project_device)(tape, self.device_projector.device, registry, observer)
        self.parity(tape, registry, observer, result)
        self.assertEqual(result.chunk.words[:, 0, 9:14].tolist(),
            [[99, 100, 6, 0, 1], [999, 1000, 6, 0, 1], [1, 100, 1, 0, 1], [1, 1000, 1, 0, 1],
             [100, 100, 0, 0, 1], [1000, 1000, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])

    def test_hidden_power_private_counterfactuals_match_host_and_pinned_source(self):
        from gpu_engine import public_context_protocol as wire
        metadata, table, dex, public, catalog = self.hp_fixture()
        handle = self.d.compile_projector(catalog, expected_binding=catalog.binding,
            expected_engine_binding=np.array(catalog.engine_binding, np.uint32))
        registry = self.device_registry(6)._replace(
            engine_binding=self.j.broadcast_to(self.j.array(catalog.engine_binding, self.j.uint32), (6, 8)))
        observer = self.observer(registry, self.j.array([0, 1, 0, 1, 0, 1], self.j.int32))
        state = self.state(6)
        state = state._replace(moves=self.j.broadcast_to(self.j.array([7, 7, 8, 8, 9, 9])[:, None, None, None], state.moves.shape),
            hidden_power_type=self.j.broadcast_to(self.j.array([17, 17, 15, 15, 10, 10])[:, None, None], state.hidden_power_type.shape),
            hidden_power_power=self.j.broadcast_to(self.j.array([70, 70, 55, 55, 63, 63])[:, None, None], state.hidden_power_power.shape))
        tape = self.e.trace_step(state, self.j.zeros((6, 2), self.j.int32), table, catalog=catalog,
            expected_binding=catalog.binding, expected_engine_binding=np.array(catalog.engine_binding, np.uint32),
            chance_inputs=self.chance(6)).callbacks
        result = self.jax.jit(self.d.project_device)(tape, handle.device, registry, observer)
        self.parity(tape, registry, observer, result, catalog)
        addresses = wire.PublicAddresses({0: wire.PublicAddress(0, 2, 17), 1: wire.PublicAddress(1, 4, 29)}, {})
        source = wire.project_line('|move|p1a: Snorlax|Hidden Power|p2a: Absol', sequence=3, turn=1,
            addresses=addresses, moves=public, abilities=self.abilities).events[0].words()
        for row in range(6):
            np.testing.assert_array_equal(result.chunk.words[row, 0], source)
        self.d.check_projector(handle)

    def test_atomic_row_refusal_overflow_unknown_payload_registry_and_namespace(self):
        j = self.j
        batch = 9
        state = self.state(batch)
        tape = self.trace(state, j.zeros((batch, 2), j.int32), chance_inputs=self.chance(batch)).callbacks
        tape = tape._replace(words=tape.words.at[1, 1, 0].set(4).at[2, 1, 8].set(7)
            .at[3, 0, 6].set(1).at[4, 1, 5].set(101),
            refusal=tape.refusal.at[5].set(self.e.OVERFLOW))
        registry = self.device_registry(batch)
        registry = registry._replace(physical_active=registry.physical_active.at[6, 0].set(2),
            engine_binding=registry.engine_binding.at[7, 0].add(j.uint32(1)))
        observer = self.observer(registry)._replace(generation=registry.generation.at[8, 1].add(1))
        result = self.jax.jit(self.d.project_device)(tape, self.device_projector.device, registry, observer)
        self.assertEqual(result.component_admitted.tolist(), [True]+[False]*8)
        self.assertEqual(result.error[0].item(), 0)
        self.assertTrue(bool((result.error[1:] != 0).all()))
        self.assertFalse(bool(result.chunk.valid[1:].any()))
        self.assertFalse(bool(result.chunk.words[1:].any()))
        np.testing.assert_array_equal(result.next_sequence[1:], observer.next_sequence[1:])
        for row in range(1, 6):
            with self.assertRaises(ValueError):
                self.e.project(tape, battle=row, stream_id=row+1, owner=0, next_sequence=3,
                    catalog=self.producer, registry=self.registry(), expected_binding=self.producer.binding,
                    expected_engine_binding=np.array(self.producer.engine_binding, np.uint32))

    def test_host_handle_guards_after_hot_jit_and_original_registry_registration(self):
        j = self.j
        public = self.p.compile_moves(self.host, self.dex, self.vocab)
        catalog = self.e.compile_catalog(self.metadata, self.engine_table, public_moves=public,
            dex=self.dex, exposure=self.exposure)
        handle = self.d.compile_projector(catalog, expected_binding=catalog.binding,
            expected_engine_binding=np.array(catalog.engine_binding, np.uint32))
        tape = self.trace(self.state(), j.zeros((1, 2), j.int32), chance_inputs=self.chance()).callbacks
        registry = self.device_registry()
        observer = self.observer(registry)
        compiled = self.jax.jit(self.d.project_device)
        compiled(tape, handle.device, registry, observer)
        with self.assertRaises(ValueError): self.d.check_projector(replace(handle))
        original = handle.device
        for value in (np.array(original.public_ids), original.public_ids.at[1].add(999)):
            object.__setattr__(handle, 'device', original._replace(public_ids=value))
            try:
                with self.assertRaises(ValueError): self.d.check_projector(handle)
            finally: object.__setattr__(handle, 'device', original)
        old = public.device
        object.__setattr__(public, 'device', old._replace(max_pp=old.max_pp.at[1].add(999)))
        try:
            compiled(tape, original, registry, observer)
            with self.assertRaises(ValueError): self.d.check_projector(handle)
        finally: object.__setattr__(public, 'device', old)
        self.assertIs(self.d.check_projector(handle), original)
        with self.assertRaises(ValueError):
            self.d.register_registry(**registry._replace(stream_id=j.array([0], j.int32))._asdict())
        for field, value in zip(registry._fields, registry):
            for readonly in (False, True):
                array = np.array(value)
                array.setflags(write=not readonly)
                with self.subTest(registry_field=field, readonly=readonly), self.assertRaises(ValueError):
                    self.d.register_registry(**registry._replace(**{field: array})._asdict())

    def test_peer_hp_and_physical_index_counterfactuals_do_not_escape_projection(self):
        j = self.j
        tape = self.e.initialize(4, capacity=1, catalog=self.producer)
        tape = self.e.condition(tape, j.ones(4, j.int32), j.array([5, 5, 1, 1], j.int32),
            j.array([799, 800, 799, 800], j.int32), j.full((4,), 1000, j.int32),
            j.zeros(4, j.int32), eligible=j.ones(4, bool))
        registry = self.device_registry(4)._replace(physical_active=j.array([[3, 5], [3, 5], [3, 1], [3, 1]], j.int32))
        observer = self.observer(registry)
        result = self.jax.jit(self.d.project_device)(tape, self.device_projector.device, registry, observer)
        self.parity(tape, registry, observer, result)
        for row in (1, 2, 3):
            np.testing.assert_array_equal(result.chunk.words[0], result.chunk.words[row])
        self.assertEqual(result.chunk.words[:, 0, 9:11].tolist(), [[80, 100]]*4)

    def test_empty_done_partial_and_sequence_exhaustion_refuses_without_wrap(self):
        j = self.j
        state = self.state()._replace(done=j.array([True]))
        tape = self.trace(state, j.zeros((1, 2), j.int32), chance_inputs=self.chance()).callbacks
        registry = self.device_registry()
        observer = self.observer(registry)
        result = self.device(tape, registry, observer)
        self.parity(tape, registry, observer, result)
        self.assertEqual(result.chunk.valid.sum().item(), 0)
        live = self.trace(self.state(), j.zeros((1, 2), j.int32), chance_inputs=self.chance()).callbacks
        result = self.device(live, registry, observer._replace(next_sequence=j.array([2**31-2], j.int32)))
        self.assertFalse(bool(result.component_admitted[0]))
        self.assertFalse(bool(result.chunk.words.any()))
        self.assertEqual(result.next_sequence.item(), 2**31-2)

    def test_actual_overflow_empty_missing_identity_self_target_and_bad_headers(self):
        j = self.j
        state, actions, chance = self.state(), j.zeros((1, 2), j.int32), self.chance()
        tape = self.trace(state, actions, chance_inputs=chance, capacity=1).callbacks
        result = self.device(tape)
        self.assertTrue(bool(result.refusal[0] & self.e.OVERFLOW))
        self.assertFalse(bool(result.component_admitted[0]))
        self.assertFalse(bool(result.chunk.valid.any()))
        self.assertFalse(bool(result.chunk.words.any()))

        # Public self targets need only the actor's prior public arrival.
        from pokezero.dex import ShowdownDex
        dex = ShowdownDex(dict(self.dex.moves, protect=replace(self.dex.moves['protect'], target='self')), {}, {})
        public = self.p.compile_moves(self.host, dex, self.vocab)
        catalog = self.e.compile_catalog(self.metadata, self.engine_table, public_moves=public, dex=dex, exposure=self.exposure)
        handle = self.d.compile_projector(catalog, expected_binding=catalog.binding,
            expected_engine_binding=np.array(catalog.engine_binding, np.uint32))
        tape = self.e.move(self.e.initialize(1, capacity=1, catalog=catalog), j.array([0]), j.array([3]),
            j.array([-999]), j.array([3]), eligible=j.array([True]))
        registry = self.device_registry()._replace(physical_active=j.array([[3, -1]], j.int32),
            public_slot=j.array([[2, -1]], j.int32), encounter=j.array([[17, 0]], j.int32),
            revealed_species_id=j.array([[11, 0]], j.int32), established=j.array([[True, False]]))
        observer = self.observer(registry)
        result = self.d.project_device(tape, handle.device, registry, observer)
        self.parity(tape, registry, observer, result, catalog)
        self.assertEqual(result.chunk.words[0, 0, 4:8].tolist(), [0, 2, 0, 2])

        tape = self.trace(state, actions, chance_inputs=chance).callbacks
        registry = self.device_registry()
        observer = self.observer(registry)
        for bad in (tape._replace(count=j.array([-1], j.int32)), tape._replace(count=j.array([65], j.int32)),
                    tape._replace(turn=j.array([-1], j.int32)), tape._replace(refusal=j.array([8], j.int32)),
                    tape._replace(engine_binding=tape.engine_binding.at[0].add(j.uint32(1))),
                    tape._replace(source_binding=tape.source_binding.at[0].add(j.uint32(1))),
                    tape._replace(move_binding=tape.move_binding.at[0].add(j.uint32(1))),
                    tape._replace(producer_binding=tape.producer_binding.at[0].add(j.uint32(1)))):
            bad_result = self.device(bad, registry, observer)
            self.assertFalse(bool(bad_result.component_admitted[0]))
            self.assertFalse(bool(bad_result.chunk.words.any()))
        for bad in (observer._replace(owner=j.array([2], j.int32)),
                    observer._replace(stream_id=j.array([99], j.int32)),
                    observer._replace(next_sequence=j.array([0], j.int32))):
            self.assertFalse(bool(self.device(tape, registry, bad).component_admitted[0]))
        with self.assertRaises(ValueError):
            self.device(tape._replace(count=j.array([4.0])), registry, observer)

    def test_compiled_device_program_contains_no_host_callback(self):
        tape = self.trace(self.state(), self.j.zeros((1, 2), self.j.int32), chance_inputs=self.chance()).callbacks
        registry = self.device_registry()
        observer = self.observer(registry)
        program = str(self.jax.make_jaxpr(self.d.project_device)(tape, self.device_projector.device, registry, observer))
        self.assertNotIn('callback', program)


# Fixture inheritance supplies helpers; avoid rerunning the thirteen original
# source tests in this module's independent device-focused test selection.
for _name in tuple(vars(fixture_module.EnginePublicEventsTests)):
    if _name.startswith('test_'):
        setattr(EnginePublicEventDeviceTests, _name, None)


if __name__ == '__main__': unittest.main()
