"""Authored Gen3 Static boundaries; not fresh-panel or game qualification."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE = os.environ.get('NODE') or shutil.which('node')
SHOWDOWN = os.environ.get('POKEZERO_SHOWDOWN_ROOT')


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated spike JAX')
class StaticTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import baton, effects, items, mechanics, phazing, pursuit, secondary, transition
        from gpu_engine.state import empty_state
        self.jax, self.j, self.m, self.tr, self.empty = jax, j, mechanics, transition, empty_state
        self.names = ['_none', 'dragonclaw', 'brickbreak', 'surf', 'pursuit', 'splash',
                      'protect', 'struggle', 'roar', 'batonpass']
        self.types = ['_none', 'normal', 'fighting', 'flying', 'poison', 'ground', 'rock',
                      'bug', 'ghost', 'steel', 'fire', 'water', 'grass', 'electric',
                      'psychic', 'ice', 'dragon', 'dark']
        self.abilities = ['_none', 'static', 'limber', 'serenegrace', 'shielddust']
        self.meta = {'vocabulary': {'moves': self.names, 'types': self.types,
            'abilities': self.abilities, 'items': ['_none', 'lumberry'], 'species': ['_none']},
            'moves': [None] + [dict(id=name, source_type='Dragon' if name == 'dragonclaw' else 'Normal',
                category='Special' if name in ('dragonclaw', 'surf', 'pursuit') else 'Physical',
                flags={'protect': 1, **({'contact': 1} if name in
                    ('dragonclaw', 'brickbreak', 'pursuit', 'struggle') else {})})
                for name in self.names[1:]]}
        n = len(self.names)
        self.table = transition.MoveTable(j.array([0, 80, 75, 95, 40, 0, 0, 50, 0, 0]),
            j.array([0, 16, 2, 11, 17, 1, 1, 0, 1, 1]),
            j.array([False, False, True, False, False, False, False, True, False, False]),
            j.full(n, 100), j.ones(n, bool), j.zeros(n, j.int32).at[self.id('protect')].set(4),
            j.ones((18, 18), j.int32), j.ones((18, 18), j.int32), j.array(self.id('struggle')),
            effects=effects.from_metadata(self.meta), items=items.from_metadata(self.meta),
            secondary=secondary.from_metadata(self.meta), pursuit=pursuit.from_metadata(self.meta),
            phazing=phazing.from_metadata(self.meta), baton=baton.from_metadata(self.meta))

    def id(self, name): return self.names.index(name)
    def ability(self, name): return self.abilities.index(name)

    def live(self, batch=60, side=0, move='dragonclaw'):
        j = self.j
        s = self.empty(batch, seed=119)
        return s._replace(hp=j.full_like(s.hp, 1000), max_hp=j.full_like(s.hp, 1000),
            stats=j.full_like(s.stats, 200).at[:, side, :, 5].set(300),
            level=j.full_like(s.level, 100), pp=j.full_like(s.pp, 10),
            moves=j.full_like(s.moves, self.id('splash')).at[:, side, 0, 0].set(self.id(move)),
            ability=s.ability.at[:, 1-side, 0].set(self.ability('static')),
            types=s.types.at[..., 0].set(self.types.index('water')), done=j.zeros_like(s.done))

    def run_case(self, state, *, side=0, table=None, switching=False, jit=False):
        j = self.j; b = state.hp.shape[0]
        c = self.tr.sample_chance_inputs(state)
        # Exhaust the actual uniform [0,60) auxiliary coordinate, independently
        # of accuracy, damage, secondaries and action-order coordinates.
        c = c._replace(phazing=j.zeros_like(c.phazing).at[:, side].set(j.arange(b) % 60))
        actions = j.zeros(state.active.shape, j.int32)
        if switching: actions = actions.at[:, 1-side].set(4)
        fn = self.jax.jit(self.tr.step_with_chance) if jit else self.tr.step_with_chance
        return fn(state, actions, self.table if table is None else table, c)

    def test_exact_one_third_on_special_contact_in_both_seats(self):
        for side in (0, 1):
            s = self.live(side=side)
            out = self.run_case(s, side=side).state
            self.assertEqual(out.status[:, side, 0].tolist(), [self.m.PAR]*20 + [0]*40)
            self.assertTrue(bool(self.j.all(out.hp[:, 1-side, 0] < 1000)))

    def test_pursuit_interception_hits_outgoing_static_and_lum_cures(self):
        for side in (0, 1):
            s = self.live(side=side, move='pursuit')
            out = self.run_case(s, side=side, switching=True).state
            self.assertEqual(out.status[:, side, 0].tolist(), [self.m.PAR]*20 + [0]*40)
            self.assertEqual(out.active[:, 1-side].tolist(), [1]*60)
            self.assertEqual(out.pp[:, side, 0, 0].tolist(), [9]*60)
            lum = s._replace(item=s.item.at[:, side, 0].set(1))
            out = self.run_case(lum, side=side, switching=True).state
            self.assertEqual(out.status[:, side, 0].tolist(), [0]*60)
            self.assertEqual(out.item[:, side, 0].tolist(), [0]*20 + [1]*40)

    def test_substitute_miss_protect_immunity_noncontact_and_no_damage_do_not_proc(self):
        s = self.live()
        cases = [(s._replace(substitute_hp=s.substitute_hp.at[:, 1, 0].set(1)), self.table),
            (s, self.table._replace(always_hit=self.table.always_hit.at[1].set(False),
                accuracy=self.table.accuracy.at[1].set(0))),
            (s._replace(moves=s.moves.at[:, 1, 0, 0].set(self.id('protect'))), self.table),
            (s, self.table._replace(type_num=self.table.type_num.at[16, 11].set(0))),
            (self.live(move='surf'), self.table)]
        for state, table in cases:
            out = self.run_case(state, table=table).state
            self.assertEqual(out.status[:, 0, 0].tolist(), [0]*60)

    def test_immunity_existing_status_dust_grace_electric_and_target_ko(self):
        for ability, status, expected in [('limber', 0, 0), ('_none', self.m.BRN, self.m.BRN),
                ('serenegrace', 0, self.m.PAR), ('shielddust', 0, self.m.PAR)]:
            s = self.live(); s = s._replace(ability=s.ability.at[:, 0, 0].set(self.ability(ability)),
                status=s.status.at[:, 0, 0].set(status), types=s.types.at[:, 0, 0, 0].set(13))
            out = self.run_case(s).state
            self.assertEqual(out.status[:20, 0, 0].tolist(), [expected]*20)
        s = self.live(); s = s._replace(hp=s.hp.at[:, 1, 0].set(1))
        out = self.run_case(s).state
        self.assertEqual(out.status[:, 0, 0].tolist(), [self.m.PAR]*20 + [0]*40)
        s = self.live(move='struggle'); s = s._replace(hp=s.hp.at[:, 0, 0].set(1))
        out = self.run_case(s).state
        self.assertEqual(out.hp[:, 0, 0].tolist(), [0]*60)
        self.assertEqual(out.status[:, 0, 0].tolist(), [self.m.PAR]*20 + [0]*40)
        lum = s._replace(item=s.item.at[:, 0, 0].set(1))
        out = self.run_case(lum).state
        self.assertEqual(out.hp[:, 0, 0].tolist(), [0]*60)
        self.assertEqual(out.status[:, 0, 0].tolist(), [0]*60)
        self.assertEqual(out.item[:, 0, 0].tolist(), [0]*20 + [1]*40)

    def test_baton_resumption_uses_saved_static_roll_against_incoming_holder(self):
        j = self.j; s = self.live(side=1)
        s = s._replace(moves=s.moves.at[:, 0, 0, 0].set(self.id('batonpass')),
            stats=s.stats.at[:, 0, :, 5].set(500),
            ability=s.ability.at[:, 0, 0].set(0).at[:, 0, 1].set(self.ability('static')))
        paused = self.run_case(s, side=1).state
        self.assertTrue(bool(j.all(paused.pending_residual)))
        self.assertEqual(paused.phazing_draws[:, 1].tolist(), list(range(60)))
        actions = j.tile(j.array([[4, 99]]), (60, 1))
        out = self.tr.step(paused, actions, self.table).state
        self.assertEqual(out.status[:, 1, 0].tolist(), [self.m.PAR]*20 + [0]*40)
        self.assertEqual(out.pp[:, 1, 0, 0].tolist(), [9]*60)
        self.assertTrue(bool(j.all(out.phazing_draws == 0)))

    def test_jit_batch_prefix_invalid_terminal_and_lum(self):
        j = self.j; s = self.live(6)
        s = s._replace(done=s.done.at[5].set(True), item=s.item.at[:3, 0, 0].set(1))
        a, b = self.run_case(s, jit=True), self.run_case(s)
        for x, y in zip(self.jax.tree.leaves(a), self.jax.tree.leaves(b)):
            self.assertTrue(bool(j.array_equal(x, y)))
        self.assertEqual(a.state.status[:3, 0, 0].tolist(), [0]*3)
        self.assertEqual(a.state.item[:3, 0, 0].tolist(), [0]*3)
        for x, y in zip(s, a.state): self.assertTrue(bool(j.array_equal(x[5:], y[5:])))
        small = self.run_case(self.jax.tree.map(lambda x: x[:3], s), jit=True)
        for x, y in zip(self.jax.tree.leaves(a), self.jax.tree.leaves(small)):
            self.assertTrue(bool(j.array_equal(x[:3], y)))
        actions = j.full(s.active.shape, 99, j.int32)
        out = self.tr.step(s, actions, self.table).state
        for x, y in zip(s, out): self.assertTrue(bool(j.array_equal(x, y)))

    @unittest.skipUnless(NODE and SHOWDOWN, 'requires pinned Showdown and Node')
    def test_pinned_gen3_contact_flags_and_actual_damaging_hit_callbacks(self):
        p = subprocess.run([NODE, str(Path(__file__).with_name('gpu_static_oracle.cjs')),
            str(Path(SHOWDOWN).resolve())], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        evidence = json.loads(p.stdout)
        self.assertEqual(evidence['static_chance'], [1, 3])
        self.assertEqual(evidence['contact'], {'dragonclaw': True, 'brickbreak': True,
            'pursuit': True, 'surf': False, 'roar': False, 'whirlwind': False})
        for name, expected in evidence['expected'].items():
            self.assertEqual(evidence['cases'][name], expected, name)


if __name__ == '__main__': unittest.main()
