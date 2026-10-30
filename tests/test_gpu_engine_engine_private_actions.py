"""Tiny literal arrays/constructor archive only; no engine init, step or game."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_private_actions as e, engine_private_team as t, engine_request_order as r
from gpu_engine import private_actions as p, source_provenance as provenance
from gpu_engine.state import BattleState
import test_gpu_engine_private_actions as fixtures


class EnginePrivateActionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.PrivateActionTests.setUpClass()
        cls.fixture = fixtures.PrivateActionTests()
        cls.catalog, cls.vocab, cls.dex = cls.fixture.catalog, cls.fixture.vocab, cls.fixture.dex
        cls.metadata = {'schema': 'literal-constructor', 'vocabulary': {
            'species': ['_none', 'pikachu', 'gengar', 'snorlax'],
            'moves': ['_none', 'surf', 'hiddenpowerice', 'curse', 'reversal', 'struggle', 'toxic', 'unmapped'],
            'types': ['_none', 'electric', 'ice', 'ghost', 'water', 'normal'],
            'abilities': ['_none', 'static', 'levitate', 'immunity'],
            'items': ['_none', 'leftovers']}}
        cls.team = t.compile_engine_private_team(cls.metadata, vocabulary=cls.vocab, dex=cls.dex,
            checkpoint_sha256='a'*64, exposure=t.RequestExposure('current', 'physical', True, True))
        cls.exposure = e.Exposure('b'*64, 'retained-constructor')
        cls.compiled = e.compile_engine_actions(cls.metadata, catalog=cls.catalog, vocabulary=cls.vocab,
            private_team=cls.team, exposure=cls.exposure)
        cls.encoder = staticmethod(e.prepare_engine_actions(cls.compiled, expected_binding=cls.compiled.device.binding))
        # The same actual immutable loader used by construction; never a real roster.
        species = np.array([[1, 2, 1, 0, 0, 0], [3, 2, 0, 0, 0, 0]], np.uint16)
        moves = np.zeros((2, 6, 4), np.uint16)
        moves[0, :3] = [1, 2, 3, 0]
        moves[1, :2] = [4, 1, 0, 0]
        pp = np.zeros_like(moves, dtype=np.int16)
        pp[0, :3] = [24, 16, 16, 0]
        pp[1, :2] = [1, 24, 0, 0]  # A noPPBoosts-like exact constructor exception.
        arrays = {name: np.zeros((2, 6), np.int32) for name in provenance._POOL_STATE_FIELDS}
        arrays.update(species=species, moves=moves, pp=pp, types=np.zeros((2, 6, 2), np.uint8),
            stats=np.zeros((2, 6, 6), np.int32), nature=np.zeros((2, 6), np.uint8),
            evs=np.zeros((2, 6, 6), np.uint16), metadata_utf8=np.frombuffer(json.dumps(cls.metadata).encode(), np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'literal.npz'
            np.savez_compressed(path, **arrays)
            cls.pool = provenance.load_constructor_pool(path)

    def ledger(self):
        o = r.initialize_order(j.array([71, 72], j.int32), j.array([0, 1], j.int32),
            j.array([3, 2], j.int32), engine_binding=j.broadcast_to(self.team.engine_binding, (2, 8)))
        receipt = r.SwitchInstalled(o.stream_id, o.owner_side, o.engine_binding, o.generation,
            j.array([True, True]), j.array([0, 0], j.int32), j.array([2, 1], j.int32))
        return r.install_switch(o, receipt)

    def constructor(self, ledger=None, indices=None):
        o = ledger or self.ledger()
        return e.retain_constructor_pp(self.pool, np.array([[0, 1], [0, 1]], np.int64) if indices is None else indices,
            stream_ids=o.stream_id, owner_sides=o.owner_side)

    def state(self):
        o = self.ledger()
        data = {field: None for field in BattleState._fields}
        for field in t.OwnedTeam._fields[2:]:
            data[field] = (j.zeros((2, 2, 6, 6), j.int32) if field == 'stats'
                else j.zeros((2, 2), j.int32) if field == 'active' else j.zeros((2, 2, 6), j.int32))
        for field in ('moves', 'pp'):
            data[field] = j.zeros((2, 2, 6, 4), j.int32)
        for field in ('hidden_power_type', 'charge_move', 'encore_move', 'encore_counter'):
            data[field] = j.zeros((2, 2, 6), j.int32)
        for row, owner in enumerate((0, 1)):
            species = j.array(self.pool.arrays['species'][row], j.int32)
            present = species > 0
            for field, values in dict(species=species, level=j.where(present, 80, 0),
                    gender=j.array([1, 2, -1, 0, 0, 0], j.int32), hp=j.where(present, 99, 0),
                    max_hp=j.where(present, 201, 0), status=j.zeros(6, j.int32),
                    item=j.where(present, 1, 0), ability=j.where(present, 1, 0),
                    base_ability=j.where(present, 1, 0), hidden_power_type=j.where(present, 2, 0)).items():
                data[field] = data[field].at[row, owner].set(values)
            stats = j.arange(36, dtype=j.int32).reshape(6, 6) + 100 + row*50
            data['stats'] = data['stats'].at[row, owner].set(j.where(present[:, None], stats, 0))
            data['moves'] = data['moves'].at[row, owner].set(j.array(self.pool.arrays['moves'][row], j.int32))
            remaining = np.array(self.pool.arrays['pp'][row], copy=True)
            remaining[remaining > 1] //= 2
            data['pp'] = data['pp'].at[row, owner].set(j.array(remaining, j.int32))
        data['active'] = data['active'].at[j.arange(2), o.owner_side].set(o.request_to_physical[:, 0])
        return BattleState(**data)

    def source(self, ledger=None, mode=e.ORDINARY):
        o = ledger or self.ledger()
        b = r.RequestBoundary(o.stream_id, o.owner_side, o.engine_binding, o.generation,
            o.request_to_physical[:, 0])
        return e.SourceRequest(b, j.broadcast_to(self.compiled.device.binding, (2, 8)),
            self.constructor(o).device.binding,
            j.array([17, 18], j.int32), j.full(2, mode, j.int32), j.ones(2, j.bool_),
            j.ones(2, j.bool_), j.zeros((2, 4), j.bool_), j.zeros(2, j.bool_), j.ones((2, 9), j.bool_))

    def encode(self, state=None, ledger=None, source=None, constructor=None):
        o = ledger or self.ledger()
        return self.encoder(state or self.state(), o, source or self.source(o), constructor or self.constructor(o),
            engine_binding=o.engine_binding)

    def raw(self, state, ledger, source, constructor):
        packets = []
        for row, owner in enumerate((0, 1)):
            party = []
            for index in ledger.request_to_physical[row].tolist():
                if index < 0: continue
                species = self.metadata['vocabulary']['species'][int(state.species[row, owner, index])]
                hp, maximum = int(state.hp[row, owner, index]), int(state.max_hp[row, owner, index])
                gender = int(state.gender[row, owner, index])
                party.append(dict(ident=f'p{owner+1}: duplicate', details=species + ', L80' + (', M' if gender == 1 else ', F' if gender == 2 else ''),
                    active=len(party) == 0, condition=f'{hp}/{maximum}' if hp else '0 fnt',
                    stats=dict(zip(('atk', 'def', 'spa', 'spd', 'spe'), state.stats[row, owner, index, 1:].tolist())),
                    ability='Static', item='Leftovers', moves=['surf', 'hiddenpowerice']))
            raw = dict(rqid=int(source.request_id[row]), side=dict(id=f'p{owner+1}', pokemon=party))
            mode = int(source.mode[row])
            if mode == e.ORDINARY:
                active = int(state.active[row, owner]); moves = []
                for slot, mid in enumerate(state.moves[row, owner, active].tolist()):
                    if not mid: break
                    name = self.metadata['vocabulary']['moves'][mid]
                    moves.append(dict(id='hiddenpower' if name.startswith('hiddenpower') else name,
                        move='Hidden Power Ice 70' if name.startswith('hiddenpower') else name,
                        pp=int(state.pp[row, owner, active, slot]), maxpp=int(constructor.device.max_pp[row, active, slot]),
                        disabled=bool(source.disabled[row, slot])))
                raw['active'] = [dict(moves=moves, trapped=bool(source.trapped[row]))]
            elif mode == e.FORCE_SWITCH: raw['forceSwitch'] = [True]
            elif mode == e.WAIT: raw['wait'] = True
            packets.append(raw)
        return packets

    def assert_matches_source(self, source=None, state=None):
        o, s = self.ledger(), state or self.state()
        source = source or self.source(o)
        constructor = self.constructor(o)
        out = self.encode(s, o, source, constructor)
        expected_facts = [p.parse_action_facts(raw, owner_side=owner, vocabulary=self.vocab, catalog=self.catalog)
            for owner, raw in enumerate(self.raw(s, o, source, constructor))]
        expected = jax.tree.map(lambda *x: j.stack(x), *expected_facts)
        self.assertEqual(out.admitted.tolist(), [True, True])
        for a, b in zip(jax.tree.leaves(out.facts), jax.tree.leaves(expected)):
            self.assertEqual(a.dtype, b.dtype)
            np.testing.assert_array_equal(a, b)
        for row, raw in enumerate(self.raw(s, o, source, constructor)):
            numeric, categories, legal = self.fixture.expected(raw, row)
            np.testing.assert_allclose(out.features.numeric[row], numeric, rtol=0, atol=1e-6)
            np.testing.assert_array_equal(out.features.categorical[row], categories)
            self.assertEqual(out.features.legal[row].tolist(), list(legal))
        return out

    def test_actual_engine_owner_arrays_reach_existing_facts_and_every_action_cell(self):
        out = self.assert_matches_source()
        self.assertEqual(out.move_action_map.tolist(), [[0, 1, 2, -1], [0, 1, -1, -1]])
        routed = e.route_choice(out, self.ledger(), j.array([4, 4], j.int32))
        self.assertEqual(routed.physical_switch_target.tolist(), [1, 0])
        self.assertEqual(routed.engine_action.tolist(), [5, 4])

    def test_forced_and_wait_have_no_invented_move_rows_or_dispatch(self):
        for mode in (e.FORCE_SWITCH, e.WAIT):
            source = self.source(mode=mode)._replace(trapped=j.ones(2, j.bool_), disabled=j.ones((2, 4), j.bool_))
            out = self.assert_matches_source(source)
            self.assertEqual(out.move_action_map.tolist(), [[-1]*4, [-1]*4])
            got = e.route_choice(out, self.ledger(), j.array([0, 0], j.int32))
            self.assertEqual(got.engine_action.tolist(), [-1, -1])
            if mode == e.WAIT:
                self.assertFalse(bool(out.features.legal.any()))
                self.assertEqual(e.route_choice(out, self.ledger(), j.array([4, 4], j.int32)).engine_action.tolist(), [-1, -1])

    def test_exact_constructor_maxpp_retained_despite_current_pp_and_base_pp(self):
        out = self.encode()
        self.assertEqual(out.facts.request.max_pp.tolist(), [[24., 16., 16., 0.], [1., 24., 0., 0.]])
        self.assertEqual(out.facts.request.pp.tolist(), [[12., 8., 8., 0.], [1., 12., 0., 0.]])
        with self.assertRaises(ValueError): self.pool.arrays['pp'].setflags(write=True)
        with self.assertRaises(ValueError): self.encode(constructor=replace(self.constructor()))
        wrong = self.constructor(indices=np.array([[1, 0], [1, 0]], np.int64))
        self.assertFalse(bool(self.encode(constructor=wrong).admitted.any()))

    def test_source_legality_not_inferred_from_zero_pp_or_server_mask(self):
        s = self.state()
        s = s._replace(pp=s.pp.at[0, 0, 2, 0].set(0))
        source = self.source()._replace(engine_legal=self.source().engine_legal.at[0, 0].set(False))
        out = self.assert_matches_source(source, s)
        self.assertTrue(bool(out.features.legal[0, 0]))
        self.assertEqual(float(out.facts.request.pp[0, 0]), 0)
        self.assertEqual(e.route_choice(out, self.ledger(), j.array([0, 0], j.int32)).engine_action.tolist(), [-1, 0])
        source = source._replace(disabled=source.disabled.at[0, 0].set(True))
        self.assertFalse(bool(self.encode(s, source=source).features.legal[0, 0]))

    def assert_refused_first(self, out):
        self.assertEqual(out.admitted.tolist(), [False, True])
        self.assertEqual(int(out.facts.request.owner_side[0]), -1)
        self.assertEqual(int(out.facts.team.owner_side[0]), -1)
        self.assertFalse(bool(out.features.numeric[0].any()))
        self.assertFalse(bool(out.features.categorical[0].any()))
        self.assertEqual(out.move_action_map[0].tolist(), [-1]*4)
        for leaf in jax.tree.leaves(out.facts.team)[2:]: self.assertFalse(np.asarray(leaf[0]).any())
        self.assertEqual(e.route_choice(out, self.ledger(), j.array([0, 0], j.int32)).engine_action.tolist(), [-1, 0])

    def test_unknown_incomplete_changed_or_compressed_requests_refuse_without_facts(self):
        source = self.source()
        for field, value in (('mode', e.UNKNOWN), ('complete', False), ('unchanged', False)):
            self.assert_refused_first(self.encode(source=source._replace(**{field: getattr(source, field).at[0].set(value)})))
        self.assert_refused_first(self.encode(source=source._replace(disabled=source.disabled.at[0].set(True))))
        for field in ('charge_move', 'encore_move', 'encore_counter'):
            s = self.state()
            self.assert_refused_first(self.encode(s._replace(**{field: getattr(s, field).at[0, 0, 2].set(1)})))

    def test_stale_active_stream_owner_namespace_and_exposure_binding_isolate_rows(self):
        source = self.source()
        for name in r.RequestBoundary._fields:
            value = getattr(source.boundary, name)
            bad = value.at[0, 0].add(j.uint32(1)) if name == 'engine_binding' else value.at[0].add(1)
            self.assert_refused_first(self.encode(source=source._replace(boundary=source.boundary._replace(**{name: bad}))))
        self.assert_refused_first(self.encode(source=source._replace(binding=source.binding.at[0, 0].add(j.uint32(1)))))
        self.assert_refused_first(self.encode(source=source._replace(constructor_binding=source.constructor_binding.at[0, 0].add(j.uint32(1)))))
        wrong_owner = r.initialize_order(self.ledger().stream_id, j.array([1, 1], j.int32),
            j.array([3, 2], j.int32), engine_binding=self.ledger().engine_binding)
        retained = self.constructor(wrong_owner)
        self.assert_refused_first(self.encode(constructor=retained))

    def test_unsupported_maps_and_bad_values_refuse_instead_of_guessing(self):
        s = self.state()
        for field, coordinate, value in (('hidden_power_type', (0, 0, 2), 0),
                ('pp', (0, 0, 2, 0), 25), ('moves', (0, 0, 2, 0), 7),
                ('moves', (0, 0, 2, 0), 5), ('active', (0, 0), -1)):
            self.assert_refused_first(self.encode(s._replace(**{field: getattr(s, field).at[coordinate].set(value)})))

    def test_each_opposing_field_is_irrelevant_to_actual_encoder_path(self):
        state = self.state(); baseline = self.encode(state)
        for field in (*t.OwnedTeam._fields[2:], *e.Selected._fields):
            poisoned = getattr(state, field).at[j.arange(2), j.array([1, 0])].set(-123456)
            out = self.encode(state._replace(**{field: poisoned}))
            for a, b in zip(jax.tree.leaves(baseline), jax.tree.leaves(out)):
                np.testing.assert_array_equal(a, b, err_msg=field)

    def test_assembly_primitive_eager_and_jit_match_without_engine_execution(self):
        state, ledger, source, retained = self.state(), self.ledger(), self.source(), self.constructor()
        owned = t.select_owned_team(state, ledger.owner_side, engine_binding=ledger.engine_binding)
        gathered = r.reorder_owned_team(owned, ledger, boundary=source.boundary)
        facts = t.team_facts(gathered.owned, self.team, expected_binding=self.team.binding)
        args = (e._select(state, ledger.owner_side), owned, facts.facts, ledger, source,
            retained.device, self.compiled.device, self.catalog.device, gathered.admitted & facts.admitted)
        eager, compiled = e._device(*args), jax.jit(e._device)(*args)
        for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(compiled)):
            self.assertEqual(a.dtype, b.dtype)
            np.testing.assert_array_equal(a, b)

    def test_original_holder_guards_every_host_call_and_outer_jit_refuses(self):
        with self.assertRaises(ValueError):
            e.prepare_engine_actions(replace(self.compiled), expected_binding=self.compiled.device.binding)
        state, ledger, source, retained = self.state(), self.ledger(), self.source(), self.constructor()
        with self.assertRaises(ValueError):
            jax.jit(lambda: self.encoder(state, ledger, source, retained, engine_binding=ledger.engine_binding))()
        source = self.source()
        for changed in (source._replace(disabled=j.zeros((2, 4), j.int32)),
                source._replace(mode=j.array([1., 1.])), source._replace(engine_legal=j.ones((2, 8), j.bool_))):
            with self.assertRaises(ValueError): self.encode(source=changed)
        catalog = replace(self.catalog, device=self.catalog.device._replace(numeric=self.catalog.device.numeric.at[1, 0, 0].add(1)))
        with self.assertRaises(ValueError):
            e.compile_engine_actions(self.metadata, catalog=catalog, vocabulary=self.vocab,
                private_team=self.team, exposure=self.exposure)
        # Guard re-runs after a previously successful cached kernel invocation.
        catalog = replace(self.catalog)
        compiled = e.compile_engine_actions(self.metadata, catalog=catalog, vocabulary=self.vocab,
            private_team=self.team, exposure=self.exposure)
        fast = e.prepare_engine_actions(compiled, expected_binding=compiled.device.binding)
        o = self.ledger()
        fast(self.state(), o, self.source(), self.constructor(), engine_binding=o.engine_binding)
        object.__setattr__(catalog, 'device', catalog.device._replace(numeric=j.array(catalog.device.numeric)))
        with self.assertRaises(ValueError): fast(self.state(), o, self.source(), self.constructor(), engine_binding=o.engine_binding)

    def test_every_static_catalog_leaf_refuses_foreign_storage_even_when_readonly(self):
        # Equal contents and copied binding words do not make foreign storage immutable.
        for field in p.ActionTables._fields:
            for writable in (True, False):
                with self.subTest(field=field, writable=writable):
                    value = np.array(getattr(self.catalog.device, field), copy=True)
                    value.setflags(write=writable)
                    catalog = replace(self.catalog, device=self.catalog.device._replace(**{field: value}))
                    with self.assertRaises(ValueError):
                        e.compile_engine_actions(self.metadata, catalog=catalog, vocabulary=self.vocab,
                            private_team=self.team, exposure=self.exposure)

    def test_mutable_numeric_cannot_change_projection_after_warmed_jit(self):
        state, ledger, source, retained = self.state(), self.ledger(), self.source(), self.constructor()
        before = self.encoder(state, ledger, source, retained, engine_binding=ledger.engine_binding)
        baseline = np.array(before.features.numeric, copy=True)  # synchronize cached execution
        numeric = np.array(self.catalog.device.numeric, copy=True)
        catalog = replace(self.catalog, device=self.catalog.device._replace(numeric=numeric))
        try:
            compiled = e.compile_engine_actions(self.metadata, catalog=catalog, vocabulary=self.vocab,
                private_team=self.team, exposure=self.exposure)
        except ValueError:
            # Rejection is the intended boundary. Mutating the rejected input cannot
            # affect the valid, already-warmed original encoder either.
            fast = self.encoder
        else:
            fast = e.prepare_engine_actions(compiled, expected_binding=compiled.device.binding)
            warm = fast(state, ledger, source, retained, engine_binding=ledger.engine_binding)
            np.testing.assert_array_equal(np.asarray(warm.features.numeric), baseline)
        numeric[catalog.ids['surf'], 0, 0] = 195
        after = fast(state, ledger, source, retained, engine_binding=ledger.engine_binding)
        self.assertEqual(after.admitted.tolist(), [True, True])
        np.testing.assert_array_equal(np.asarray(after.features.numeric), baseline)

    def test_caller_registration_mutation_cannot_rebind_warmed_constructor(self):
        for field in ('stream_id', 'owner_side'):
            with self.subTest(field=field):
                state, ledger, source = self.state(), self.ledger(), self.source()
                streams = np.array(ledger.stream_id, copy=True)
                owners = np.array(ledger.owner_side, copy=True)
                retained = e.retain_constructor_pp(self.pool, np.array([[0, 1], [0, 1]], np.int64),
                    stream_ids=streams, owner_sides=owners)
                binding = np.array(retained.device.binding, copy=True)
                warm = self.encoder(state, ledger, source, retained, engine_binding=ledger.engine_binding)
                baseline = np.array(warm.features.numeric, copy=True)
                self.assertEqual(warm.admitted.tolist(), [True, True])
                if field == 'stream_id':
                    streams[0] = 171
                    attempted_ledger = ledger._replace(stream_id=j.array(streams, j.int32))
                    attempted_state = state
                else:
                    owners[0] = 1
                    attempted_ledger = ledger._replace(owner_side=j.array(owners, j.int32))
                    # A plausible owning physical packet prevents empty opponent
                    # storage from incidentally hiding the registration defect.
                    changed = {name: value.at[0, 1].set(value[0, 0])
                        for name, value in zip(BattleState._fields, state) if value is not None}
                    attempted_state = state._replace(**changed)
                attempted_source = source._replace(boundary=source.boundary._replace(
                    stream_id=attempted_ledger.stream_id, owner_side=attempted_ledger.owner_side))
                attempted = self.encoder(attempted_state, attempted_ledger, attempted_source, retained,
                    engine_binding=attempted_ledger.engine_binding)
                self.assertEqual(attempted.admitted.tolist(), [False, True])
                np.testing.assert_array_equal(retained.device.binding, binding)
                np.testing.assert_array_equal(retained.device.stream_id, ledger.stream_id)
                np.testing.assert_array_equal(retained.device.owner_side, ledger.owner_side)
                original = self.encoder(state, ledger, source, retained, engine_binding=ledger.engine_binding)
                self.assertEqual(original.admitted.tolist(), [True, True])
                np.testing.assert_array_equal(original.features.numeric, baseline)

    def test_all_retained_static_arrays_have_immutable_jax_storage(self):
        ledger = self.ledger()
        retained = e.retain_constructor_pp(self.pool, np.array([[0, 1], [0, 1]], np.int64),
            stream_ids=np.array(ledger.stream_id, copy=True), owner_sides=np.array(ledger.owner_side, copy=True))
        for name, arrays in (('action catalog', self.catalog.device), ('adapter tables', self.compiled.device),
                ('constructor', retained.device), ('private team', self.team.device)):
            for field, value in zip(arrays._fields, arrays):
                with self.subTest(holder=name, field=field):
                    self.assertIsInstance(value, jax.Array)
                    view = np.asarray(value)
                    self.assertFalse(view.flags.writeable)
                    with self.assertRaises(ValueError): view.setflags(write=True)


if __name__ == '__main__': unittest.main()
