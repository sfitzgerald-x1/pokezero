"""Authored suspension correspondence, not fresh-root or pool fidelity credit.

Requires an isolated `conditioned-oracle` search wheel and full-PP binding.
The historical mapper and every historical panel remain unchanged.
"""
from copy import deepcopy
import importlib.util
import json
import unittest

AVAILABLE = all(importlib.util.find_spec(m) for m in ("jax", "poke_engine", "pokezero_search"))
if AVAILABLE:
    import pokezero_search
    AVAILABLE = bool(getattr(pokezero_search, "CONDITIONED_ORACLE_ENABLED", False))


@unittest.skipUnless(AVAILABLE, "requires isolated conditioned-oracle native wheels and JAX")
class ConditionedResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        import poke_engine
        from gpu_engine import baton, future_state as f, items, secondary, transition as t
        from gpu_engine.state import empty_state
        cls.jax, cls.j, cls.engine, cls.f, cls.t = jax, j, poke_engine, f, t
        names = ["_none", "batonpass", "thunderbolt", "splash", "struggle", "bodyslam"]
        types = ["_none", "normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost",
            "steel", "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark"]
        cls.meta = {"vocabulary": {"moves": names, "types": types,
            "species": ["_none", "ditto"], "abilities": ["_none", "static"],
            "items": ["_none", "leftovers"]},
            "moves": [None] + [{"id": name, "source_type": "Electric" if name == "thunderbolt" else "Normal",
                "category": "Special" if name == "thunderbolt" else "Physical" if name == "bodyslam" else "Status",
                "flags": {"protect": 1, "contact": int(name == "bodyslam")}} for name in names[1:]]}
        cls.table = t.MoveTable(j.array([0, 0, 95, 0, 50, 85]), j.array([0, 1, 13, 1, 1, 1]),
            j.array([False, False, False, False, True, True]), j.full(6, 100),
            j.array([True, True, False, True, False, False]), j.zeros(6, j.int32),
            j.ones((18, 18), j.int32).at[13, 5].set(0).at[1, 8].set(0),
            j.ones((18, 18), j.int32).at[13, 12].set(2), j.array(4),
            baton=baton.from_metadata(cls.meta), items=items.from_metadata(cls.meta),
            secondary=secondary.from_metadata(cls.meta))
        cls.empty = staticmethod(empty_state)
        cls.step = staticmethod(jax.jit(t.step_with_chance))

    def fixture(self, replacing, secondary_roll, boundary=None, *, contact_roll=59, queued_move="thunderbolt", static_at=None, poison=True):
        j = self.j
        state = self.empty(1, seed=771)._replace(
            hp=j.full((1, 2, 6), 1000), max_hp=j.full((1, 2, 6), 1000),
            stats=j.full((1, 2, 6, 6), 200).at[..., 0].set(1000),
            species=j.ones((1, 2, 6), j.int32), level=j.full((1, 2, 6), 100),
            types=j.zeros((1, 2, 6, 2), j.int32).at[..., 0].set(1),
            moves=j.full((1, 2, 6, 4), 3), pp=j.full((1, 2, 6, 4), 10),
            done=j.array([False]))
        queued = 1 - replacing
        state = state._replace(
            stats=state.stats.at[:, replacing, :, 5].set(300)
                .at[:, replacing, 1, 5].set(200).at[:, queued, :, 5].set(101),
            moves=state.moves.at[:, replacing, 0, 0].set(1).at[:, queued, 0, 0].set(self.meta["vocabulary"]["moves"].index(queued_move)),
            status=state.status.at[:, queued, 0].set(3 if poison else 0),
            hp=state.hp.at[:, replacing, 1].set(800),
            item=state.item.at[:, replacing, 1].set(1))
        draws = j.zeros((1, 2, 12), j.int32).at[:, :, 4].set(15).at[:, :, 5].set(7)
        chance = self.t.ChanceInputs(draws, j.full((1, 2), contact_roll),
            j.full((1, 2), secondary_roll), j.full((1, 2), 59),
            j.array([[0, 1]]), j.array([0]))
        if boundary == "terminal_poison":
            state = state._replace(hp=state.hp.at[:, queued, :].set(0).at[:, queued, 0].set(100))
        elif boundary in ("minimum_damage", "immunity"):
            state = state._replace(
                stats=state.stats.at[:, queued, 0, 3].set(1)
                    .at[:, replacing, 1, 4].set(30000),
                types=state.types.at[:, replacing, 1, 0].set(12 if boundary == "minimum_damage" else 5),
                item=state.item.at[:, replacing, 1].set(0))
            chance = chance._replace(draws=chance.draws.at[:, :, 5].set(0))
        if static_at is not None:
            state = state._replace(ability=state.ability.at[:, replacing, static_at].set(1))
        if boundary == "ghost_contact":
            state = state._replace(types=state.types.at[:, replacing, 1, 0].set(8))
        if queued_move == "splash":
            state = state._replace(stats=state.stats.at[:, queued, :, 5].set(501))
        native = self.f.native_input_from_jax(state, 0, self.meta, module=self.engine)
        self.assertIsInstance(native, self.f.NativeInput, native)
        branches = self.engine.generate_instructions(native.state, "batonpass" if replacing == 0 else queued_move,
            queued_move if replacing == 0 else "batonpass")
        self.assertEqual(len(branches), 1)
        paused_native = native.state.apply_instructions(branches[0])
        paused = self.step(state, j.zeros((1, 2), j.int32), self.table, chance).state
        self.assertTrue(bool(paused.pending_residual[0]))
        self.assertEqual(int(paused.pp[0, queued, 0, 0]), 9 if queued_move == "splash" else 10)
        self.assertEqual(int(paused.hp[0, queued, 0]), int(state.hp[0, queued, 0]),
            "poison must not tick before replacement")
        # The general mapper must still refuse this unqualified domain.
        refused = self.f.native_input_from_jax(paused, 0, self.meta, module=self.engine)
        self.assertIsInstance(refused, self.f.Refused)
        self.assertIn("LIVE_SUSPENSION_INTERFACE", {g.code for g in refused.gaps})
        data = self.f._sample(paused, 0)
        capsule = {k: data[k] for k in ("pending_residual", "forced_switch", "baton_pass", "queued_actor",
            "queued_actions", "queued_stall", "residual_draws", "phazing_draws", "secondary_draws",
            "sleep_talk_draws", "replacement_count", "turn", "keys")}
        capsule.update(version=1, schema="pokezero.suspended-turn")
        return paused, paused_native, native.certificate, capsule, chance

    def test_full_successors_both_seats_saved_secondary_and_no_redraw(self):
        j = self.j
        for replacing in (0, 1):
            outcomes = []
            for roll in (9, 10):
                paused, native, certificate, capsule, chance = self.fixture(replacing, roll)
                before_native, before_capsule = native.to_string(), deepcopy(capsule)
                report = json.loads(pokezero_search.conditioned_resume(before_native, json.dumps(capsule), 1, [0, 1]))
                self.assertEqual(len(report["branches"]), 1, "no fresh secondary or damage mixture")
                branch = report["branches"][0]
                self.assertEqual(branch["percentage"], 100)
                # Ignored foe action is deliberately slot 3, not queued slot 0.
                actions = j.array([[4, 3] if replacing == 0 else [3, 4]])
                expected = None
                for new_roll in (0, 99):
                    new_chance = chance._replace(
                        draws=j.zeros_like(chance.draws).at[:, :, 4].set(0)
                            .at[:, :, 5].set(15).at[:, :, 6].set(new_roll),
                        secondary=j.full_like(chance.secondary, new_roll))
                    result = self.step(paused, actions, self.table, new_chance)
                    self.assertFalse(bool(result.invalid_actions[0]))
                    actual = self.f.canonical_jax_future(result.state, 0, self.meta, certificate=certificate)
                    self.assertIsInstance(actual, self.f.CanonicalFuture, actual)
                    oracle = self.f.canonical_native_future(branch["state"], certificate)
                    self.assertIsInstance(oracle, self.f.CanonicalFuture, oracle)
                    self.assertEqual(self.f.field_differences(actual.key, oracle.key), ())
                    if expected is not None:
                        for a, b in zip(self.jax.tree.leaves(expected), self.jax.tree.leaves(result.state)):
                            self.assertTrue(bool(j.array_equal(a, b)), "new ordinary draws changed resume")
                    expected = result.state
                    out_data = self.f._sample(result.state, 0)
                    for name in capsule.keys() - {"version", "schema"}:
                        self.assertEqual(json.loads(json.dumps(out_data[name])), branch["capsule"][name], name)
                outcomes.append(int(expected.status[0, replacing, 1]))
                self.assertEqual(int(expected.pp[0, 1-replacing, 0, 0]), 9)
                self.assertEqual(int(expected.hp[0, 1-replacing, 0]), 875, "exactly one poison tick")
                self.assertEqual(native.to_string(), before_native)
                self.assertEqual(capsule, before_capsule)
            self.assertEqual(outcomes, [1, 0], "saved 9/10 boundary must cause paralysis/no paralysis")

    def test_bad_capsules_and_unsupported_context_refuse_without_fallback(self):
        _, native, _, capsule, _ = self.fixture(0, 9)
        for mutation in (lambda c: c.update(secondary_draws=[100, 9]),
                         lambda c: c.update(queued_actions=[0, 3]),
                         lambda c: c.update(replacement_count=1),
                         lambda c: c.update(new_draws=[0, 0])):
            bad = deepcopy(capsule); mutation(bad)
            with self.assertRaises(ValueError):
                pokezero_search.conditioned_resume(native.to_string(), json.dumps(bad), 1, [0, 1])

    def test_terminal_poison_and_minimum_damage_full_successors(self):
        for replacing in (0, 1):
            for boundary in ("terminal_poison", "minimum_damage", "immunity"):
                with self.subTest(replacing=replacing, boundary=boundary):
                    paused, native, certificate, capsule, chance = self.fixture(replacing, 99, boundary)
                    actions = self.j.array([[4, 3] if replacing == 0 else [3, 4]])
                    expected = self.step(paused, actions, self.table, chance).state
                    report = json.loads(pokezero_search.conditioned_resume(
                        native.to_string(), json.dumps(capsule), 1, [0, 1]))
                    self.assertEqual(len(report["branches"]), 1)
                    branch = report["branches"][0]
                    actual = self.f.canonical_jax_future(expected, 0, self.meta, certificate=certificate)
                    oracle = self.f.canonical_native_future(branch["state"], certificate)
                    self.assertIsInstance(actual, self.f.CanonicalFuture, actual)
                    self.assertIsInstance(oracle, self.f.CanonicalFuture, oracle)
                    self.assertEqual(self.f.field_differences(actual, oracle), ())
                    out_data = self.f._sample(expected, 0)
                    for name in capsule.keys() - {"version", "schema"}:
                        self.assertEqual(json.loads(json.dumps(out_data[name])), branch["capsule"][name], name)
                    if boundary == "terminal_poison":
                        self.assertTrue(bool(expected.done[0]))
                        self.assertEqual(branch["capsule"]["forced_switch"], [False, False])
                    else:
                        self.assertEqual(int(expected.hp[0, replacing, 1]),
                            799 if boundary == "minimum_damage" else 800)

    def test_saved_static_and_target_secondary_coordinates_are_independent(self):
        for replacing in (0, 1):
            cases = [(r, s, 1, False, None) for r in (19, 20) for s in (29, 30)]
            cases += [(19, 29, None, False, None), (19, 29, 0, False, None),
                      (19, 29, 1, True, None), (19, 29, 1, False, "ghost_contact")]
            for contact, secondary, static_at, poison, boundary in cases:
                with self.subTest(replacing=replacing, contact=contact, secondary=secondary,
                                 static_at=static_at, poison=poison, boundary=boundary):
                    paused, native, certificate, capsule, chance = self.fixture(replacing, secondary,
                        boundary, contact_roll=contact, queued_move="bodyslam", static_at=static_at, poison=poison)
                    # Saved actor coordinate differs from the holder's, too.
                    queued = 1-replacing
                    capsule["phazing_draws"] = list(capsule["phazing_draws"])
                    capsule["phazing_draws"][replacing] = 59-contact
                    paused = paused._replace(phazing_draws=paused.phazing_draws.at[:, replacing].set(59-contact))
                    actions = self.j.array([[4,3] if replacing == 0 else [3,4]])
                    branches = json.loads(pokezero_search.conditioned_resume(native.to_string(),
                        json.dumps(capsule),1,[0,1]))["branches"]
                    self.assertEqual(len(branches),1)
                    branch = branches[0]
                    results = []
                    for fresh in (0,59):
                        new = chance._replace(phazing=self.j.full_like(chance.phazing,fresh),
                            secondary=self.j.full_like(chance.secondary,0 if fresh else 99),
                            draws=self.j.zeros_like(chance.draws).at[:,:,4].set(0).at[:,:,5].set(15))
                        expected = self.step(paused,actions,self.table,new).state
                        actual = self.f.canonical_jax_future(expected,0,self.meta,certificate=certificate)
                        oracle = self.f.canonical_native_future(branch["state"],certificate)
                        self.assertIsInstance(actual,self.f.CanonicalFuture,actual)
                        self.assertIsInstance(oracle,self.f.CanonicalFuture,oracle)
                        self.assertEqual(self.f.field_differences(actual,oracle),())
                        out = self.f._sample(expected,0)
                        for name in capsule.keys()-{"version","schema"}:
                            self.assertEqual(json.loads(json.dumps(out[name])),branch["capsule"][name],name)
                        self.assertEqual(int(expected.pp[0,queued,0,0]),9)
                        ghost = boundary == "ghost_contact"
                        self.assertEqual(int(expected.status[0,replacing,1]),int(secondary<30 and not ghost))
                        self.assertEqual(int(expected.status[0,queued,0]),3 if poison else int(contact<20 and static_at==1 and not ghost))
                        if ghost:self.assertEqual(int(expected.hp[0,replacing,1]),862)
                        results.append(expected)
                    for a,b in zip(self.jax.tree.leaves(results[0]),self.jax.tree.leaves(results[1])):
                        self.assertTrue(bool(self.j.array_equal(a,b)))

    def test_second_action_baton_runs_only_deferred_residual(self):
        for replacing in (0,1):
            paused,native,certificate,capsule,chance=self.fixture(replacing,9,queued_move="splash")
            self.assertEqual(int(paused.queued_actor[0]),-1)
            self.assertEqual(paused.queued_actions.tolist(),[[0,0]])
            before=native.to_string()
            branches=json.loads(pokezero_search.conditioned_resume(before,json.dumps(capsule),1,[0,1]))["branches"]
            self.assertEqual(len(branches),1)
            branch=branches[0]
            self.assertEqual(branch["percentage"],100)
            for ignored in (0,3):
                for fresh in (0,59):
                    actions=self.j.array([[4,ignored] if replacing==0 else [ignored,4]])
                    new=chance._replace(phazing=self.j.full_like(chance.phazing,fresh),
                        secondary=self.j.full_like(chance.secondary,fresh),
                        draws=self.j.zeros_like(chance.draws).at[:,:,4].set(0).at[:,:,5].set(15))
                    result=self.step(paused,actions,self.table,new)
                    self.assertFalse(bool(result.invalid_actions[0]))
                    actual=self.f.canonical_jax_future(result.state,0,self.meta,certificate=certificate)
                    oracle=self.f.canonical_native_future(branch["state"],certificate)
                    self.assertIsInstance(actual,self.f.CanonicalFuture,actual)
                    self.assertIsInstance(oracle,self.f.CanonicalFuture,oracle)
                    self.assertEqual(self.f.field_differences(actual,oracle),())
                    out=self.f._sample(result.state,0)
                    for name in capsule.keys()-{"version","schema"}:
                        self.assertEqual(json.loads(json.dumps(out[name])),branch["capsule"][name],name)
                    self.assertEqual(result.state.pp[0,1-replacing,0].tolist(),[9,10,10,10])
                    self.assertEqual(int(result.state.hp[0,1-replacing,0]),875)
                    self.assertEqual(int(result.state.hp[0,replacing,1]),862)
            self.assertEqual(native.to_string(),before)
            # Do not repair a contradictory parent by erasing its real saved move.
            _,first_native,_,first_capsule,_=self.fixture(replacing,9)
            first_capsule["queued_actor"]=-1
            with self.assertRaisesRegex(ValueError,"UNSUPPORTED_EMPTY_QUEUE_CONTEXT"):
                pokezero_search.conditioned_resume(first_native.to_string(),json.dumps(first_capsule),1,[0,1])


if __name__ == "__main__":
    unittest.main()
