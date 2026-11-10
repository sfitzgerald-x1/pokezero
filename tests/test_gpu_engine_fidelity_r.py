"""Fidelity patch R: the appended effect chance stream and Bonemerang's second hit, on authored states.

The legacy-stream equality test (design 2.1) checks the six original ChanceInputs fields and the key advance
against a frozen copy of the pre-R sampler. The Bonemerang checks follow the Showdown f76228a1 rules cited in
gpu_engine/transition.py; the two-engine probes (gpu_engine/effect_probes/bonemerang*.json) are the Showdown gate.
Also: the R regression plug-in and the A2-review carries L1/L2 of gpu_engine/v2e_regress.py.
Every ChanceInputs here is built with keywords.
"""
import importlib.util
import unittest

import numpy as np

MOVES = ['_none', 'bonemerang', 'tackle', 'splash', 'substitute', 'counter', 'destinybond', 'endure', 'protect',
         'struggle', 'sleeptalk', 'leechseed']
(BONEMERANG, TACKLE, SPLASH, SUBSTITUTE, COUNTER, DESTINYBOND, ENDURE, PROTECT, STRUGGLE, SLEEPTALK, LEECHSEED) = range(1, 12)
TYPES = ['_none', 'normal', 'fighting', 'flying', 'poison', 'ground', 'rock', 'bug', 'ghost',
         'steel', 'fire', 'water', 'grass', 'electric', 'psychic', 'ice', 'dragon', 'dark']
NORMAL, FIGHTING, FLYING, GROUND, GHOST = 1, 2, 3, 5, 8
ABILITIES = ['_none', 'battlearmor', 'levitate', 'liquidooze', 'colorchange']
BATTLEARMOR, LEVITATE, LIQUIDOOZE, COLORCHANGE = 1, 2, 3, 4
ROCK = 6
B = 8
POWER = {BONEMERANG: 50, TACKLE: 35, STRUGGLE: 50}
TYPE = {BONEMERANG: GROUND, COUNTER: FIGHTING, DESTINYBOND: GHOST}
PHYSICAL = {BONEMERANG, TACKLE, STRUGGLE, COUNTER}
ALWAYS = {SPLASH, SUBSTITUTE, DESTINYBOND, ENDURE, PROTECT, STRUGGLE, SLEEPTALK, LEECHSEED}
ACCURACY = {BONEMERANG: 90}
PRIORITY = {COUNTER: -5, PROTECT: 3, ENDURE: 4}
PROTECTABLE = {BONEMERANG, TACKLE, COUNTER, STRUGGLE}
CONTACT = {TACKLE, COUNTER, STRUGGLE}
BYPASS = {DESTINYBOND}
HAS_JAX = bool(importlib.util.find_spec('jax'))


def metadata():
    rows = [None] + [{'id': name, 'source_type': 'Normal', 'category': 'Physical' if i in PHYSICAL else 'Status',
                      'flags': {'protect': int(i in PROTECTABLE), 'contact': int(i in CONTACT), 'bypasssub': int(i in BYPASS)}}
                     for i, name in enumerate(MOVES) if i]
    return {'vocabulary': {'moves': MOVES, 'types': TYPES, 'abilities': ABILITIES, 'items': ['_none'],
                           'species': ['_none', 'a', 'b', 'c', 'd', 'e', 'f']}, 'moves': rows}


def components():
    import jax.numpy as j
    from gpu_engine import critical, damage_moves, effects, items, secondary, sleep_talk, sustain, transition, variable_moves
    meta = metadata();n = len(MOVES)
    num = np.ones((18, 18), np.int32);num[GROUND, FLYING] = 0;num[NORMAL, GHOST] = 0;num[FIGHTING, GHOST] = 0;num[GROUND, ROCK] = 2
    table = transition.MoveTable(
        j.asarray([POWER.get(i, 0) for i in range(n)], j.int32), j.asarray([TYPE.get(i, NORMAL) for i in range(n)], j.int32),
        j.asarray([i in PHYSICAL for i in range(n)]), j.asarray([ACCURACY.get(i, 100) for i in range(n)], j.int32),
        j.asarray([i in ALWAYS for i in range(n)]), j.asarray([PRIORITY.get(i, 0) for i in range(n)], j.int32),
        j.asarray(num), j.ones((18, 18), j.int32), j.array(STRUGGLE),
        effects=effects.from_metadata(meta), items=items.from_metadata(meta), damage_moves=damage_moves.from_metadata(meta),
        variable_moves=variable_moves.from_metadata(meta), sleep_talk=sleep_talk.from_metadata(meta),
        critical=critical.from_metadata(meta), sustain=sustain.from_metadata(meta), secondary=secondary.from_metadata(meta))
    return meta, table


def frozen_sample_chance_inputs(state):
    """transition.sample_chance_inputs exactly as it was before fidelity R (base 188de7a3), six fields."""
    import jax
    import jax.numpy as jnp
    keys = jax.vmap(lambda key:jax.random.split(key,2))(state.keys)[:,1]
    draw = lambda salt,limit:jax.vmap(lambda key:jax.random.randint(
        jax.random.fold_in(key,salt),(2,),0,limit))(keys)
    return (
        jax.vmap(lambda key:jax.random.randint(key,(2,12),0,
            jnp.array([2,100,100,100,16,16,100,4,8,2,2,2],jnp.int32)))(keys),
        draw(271828,60),draw(1618033,100),draw(1414213,60),
        jax.vmap(lambda key,count:jax.random.randint(jax.random.fold_in(
            key,jnp.uint32(982451653)+count.astype(jnp.uint32)),(2,),0,2))(
                state.keys,state.replacement_count),
        jax.vmap(lambda key:jax.random.randint(jax.random.fold_in(key,314159),(),0,2))(state.keys))


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class LegacyStreamTests(unittest.TestCase):
    """Design 2.1: the six original fields and the key advance are bitwise unchanged; effect is appended."""
    LANES = 4096

    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import transition
        from gpu_engine.state import empty_state
        cls.jax, cls.j, cls.t = jax, j, transition
        rng = np.random.default_rng(20261110)
        keys = rng.integers(0, 1 << 32, (cls.LANES, 2), dtype=np.uint64).astype(np.uint32)
        keys[:4] = [[0, 0], [0, 1], [0xFFFFFFFF, 0xFFFFFFFF], [1732050, 271828]]
        counts = rng.integers(0, 64, cls.LANES).astype(np.int32)
        cls.state = empty_state(cls.LANES)._replace(keys=j.asarray(keys), replacement_count=j.asarray(counts))

    def test_six_legacy_fields_are_bitwise_equal_to_the_frozen_sampler(self):
        new = self.jax.jit(self.t.sample_chance_inputs)(self.state)
        old = self.jax.jit(frozen_sample_chance_inputs)(self.state)
        self.assertEqual(self.t.ChanceInputs._fields[:6], ('draws', 'phazing', 'secondary', 'sleep_talk', 'entry_ties', 'initial_tie'))
        self.assertEqual(self.t.ChanceInputs._fields[6], 'effect')
        for name, value in zip(self.t.ChanceInputs._fields[:6], old):
            np.testing.assert_array_equal(np.asarray(getattr(new, name)), np.asarray(value), err_msg=name)
            self.assertEqual(getattr(new, name).dtype, value.dtype, name)

    def test_effect_is_its_own_fold_in_of_the_split_key(self):
        jax, j = self.jax, self.j
        new = self.t.sample_chance_inputs(self.state)
        split = jax.vmap(lambda k: jax.random.split(k, 2))(self.state.keys)[:, 1]
        expected = jax.vmap(lambda k: jax.random.randint(jax.random.fold_in(k, 1732050), (2, 4), 0,
            j.array([16, 16, 4, 2], j.int32)))(split)
        effect = np.asarray(new.effect)
        np.testing.assert_array_equal(effect, np.asarray(expected))
        self.assertEqual(effect.shape, (self.LANES, 2, 4));self.assertEqual(new.effect.dtype, j.int32)
        for column, bound in enumerate((16, 16, 4, 2)):
            self.assertEqual(set(np.unique(effect[..., column]).tolist()), set(range(bound)))
        # A different stream from every original field it could alias (same split key, other salts).
        self.assertFalse(np.array_equal(effect[..., 0], np.asarray(new.draws)[..., 4]))
        self.assertEqual((self.t.EFFECT_SALT, self.t.EFFECT_BOUNDS), (1732050, (16, 16, 4, 2)))

    def test_key_advance_and_six_field_callers_are_unchanged(self):
        jax, j, t = self.jax, self.j, self.t
        import test_gpu_engine_chance_inputs as fixture
        _, table = fixture.fixture(1)
        state = self.state
        state = state._replace(stats=j.full_like(state.stats, 200).at[..., 0].set(100), level=j.full_like(state.level, 100),
            hp=j.full_like(state.hp, 100), max_hp=j.full_like(state.hp, 100), moves=j.ones_like(state.moves),
            pp=j.full_like(state.pp, 10), types=state.types.at[..., 0].set(1),
            done=j.asarray(np.arange(self.LANES) % 7 == 3))
        actions = j.zeros((self.LANES, 2), j.int32)
        out = jax.jit(lambda s, a: t.step(s, a, table))(state, actions).state
        split0 = np.asarray(jax.vmap(lambda k: jax.random.split(k, 2))(state.keys)[:, 0])
        done = np.asarray(state.done)
        np.testing.assert_array_equal(np.asarray(out.keys)[~done], split0[~done])
        np.testing.assert_array_equal(np.asarray(out.keys)[done], np.asarray(state.keys)[done])
        # A caller that omits `effect` (the six positional fields) gets exactly the sampled trajectory.
        sampled = t.sample_chance_inputs(state)
        six = t.ChanceInputs(draws=sampled.draws, phazing=sampled.phazing, secondary=sampled.secondary,
            sleep_talk=sampled.sleep_talk, entry_ties=sampled.entry_ties, initial_tie=sampled.initial_tie)
        self.assertIsNone(six.effect)
        step = jax.jit(lambda s, a, c: t.step_with_chance(s, a, table, c))
        for left, right in zip(jax.tree.leaves(step(state, actions, six)), jax.tree.leaves(step(state, actions, sampled)), strict=True):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


@unittest.skipUnless(HAS_JAX, 'requires JAX')
class BonemerangTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import transition
        from gpu_engine.state import BattleState, DESTINY_BOND, empty_state
        cls.jax, cls.j, cls.t, cls.State, cls.DB = jax, j, transition, BattleState, DESTINY_BOND
        cls.empty = staticmethod(empty_state)
        cls.meta, cls.table = components()
        single = cls.table.damage_moves._replace(hits=j.ones_like(cls.table.damage_moves.hits))
        cls.single = cls.table._replace(damage_moves=single)     # the same move as one hit, for reference damage
        cls.step = staticmethod(jax.jit(transition.step_with_chance))

    # -- authored batch: B lanes, side 0 faster, both actives at full 300 HP, side 0 Bonemerang, side 1 Splash.
    def build(self, lanes):
        j = self.j;s = self.empty(B, seed=4242)
        s = s._replace(hp=j.full_like(s.hp, 300), max_hp=j.full_like(s.max_hp, 300),
            stats=j.full_like(s.stats, 200).at[..., 0].set(300).at[:, 0, :, 5].set(301),
            level=j.full_like(s.level, 100), types=s.types.at[..., 0].set(NORMAL), base_types=s.base_types.at[..., 0].set(NORMAL),
            species=j.broadcast_to(j.arange(1, 7, dtype=j.int32), (B, 2, 6)),
            moves=j.full_like(s.moves, SPLASH).at[:, 0, 0, 0].set(BONEMERANG), pp=j.full_like(s.pp, 10), done=j.zeros(B, bool))
        arrays = {k: np.array(v) for k, v in s._asdict().items()}
        for lane, edits in enumerate(lanes):
            for field, writes in edits.items():
                for index, value in writes:arrays[field][(lane,) + tuple(index)] = value
        return self.State(**{k: self.j.asarray(v) for k, v in arrays.items()})

    def chance(self, draws=(), effect=()):
        """No thaw/full paralysis/confusion hit, no crit, max damage roll, accuracy roll 0, side 0 wins ties; the
        effect stream: no second-hit crit, max second-hit roll. draws/effect: per-lane {(side, column): value}."""
        j = self.j;d = np.zeros((B, 2, 12), np.int32);d[:, :, 1:3] = 99;d[:, :, 4] = 15;d[:, :, 5] = 15
        e = np.zeros((B, 2, 4), np.int32);e[:, :, 0] = 15;e[:, :, 1] = 15
        for target, lanes in ((d, draws), (e, effect)):
            for lane, edits in enumerate(lanes):
                for (side, column), value in edits.items():target[lane, side, column] = value
        z = j.zeros((B, 2), j.int32)
        return self.t.ChanceInputs(draws=j.asarray(d), phazing=z, secondary=j.full((B, 2), 99, j.int32), sleep_talk=z,
                                   entry_ties=z, initial_tie=j.zeros(B, j.int32), effect=j.asarray(e))

    def play(self, s, chance=None, table=None, a0=0, a1=0):
        actions = self.j.zeros((B, 2), self.j.int32).at[:, 0].set(a0).at[:, 1].set(a1)
        out = self.step(s, actions, self.table if table is None else table, self.chance() if chance is None else chance)
        self.assertFalse(bool(np.asarray(out.invalid_actions).any()))
        return out.state

    @staticmethod
    def merge(*parts):
        out = {}
        for part in parts:
            for k, v in part.items():out.setdefault(k, []).extend(v)
        return out

    def single_hit(self, s, crit, roll):
        """[B] damage of ONE Bonemerang hit with (crit draw, roll draw), from the engine itself."""
        out = self.play(s, self.chance(draws=[{(0, 4): crit[k], (0, 5): roll[k]} for k in range(B)]), table=self.single)
        return np.asarray(s.hp)[:, 1, 0] - np.asarray(out.hp)[:, 1, 0]

    def test_tables(self):
        hits = np.asarray(self.table.damage_moves.hits)
        self.assertEqual(hits.tolist(), [2 if i == BONEMERANG else 1 for i in range(len(MOVES))])

    def test_each_hit_rolls_its_own_critical_and_damage(self):
        s = self.build([{}] * B)
        combos = [(c1, r1, c2, r2) for c1 in (15, 0) for c2 in (15, 0) for r1, r2 in ((0, 15), (15, 0))]
        out = self.play(s, self.chance(draws=[{(0, 4): c1, (0, 5): r1} for c1, r1, _, _ in combos],
                                       effect=[{(0, 0): c2, (0, 1): r2} for _, _, c2, r2 in combos]))
        first = self.single_hit(s, [c[0] for c in combos], [c[1] for c in combos])
        second = self.single_hit(s, [c[2] for c in combos], [c[3] for c in combos])
        taken = 300 - np.asarray(out.hp)[:, 1, 0]
        np.testing.assert_array_equal(taken, first + second)
        self.assertGreater(int(second[2]), int(second[0]))                  # crit2 (effect[0][0]=0) doubles the second hit
        self.assertNotEqual(int(first[0]), int(first[1]))                   # the rolls differ
        self.assertTrue((np.asarray(out.hp)[:, 1, 0] > 0).all())
        self.assertEqual(np.asarray(out.pp)[:, 0, 0, 0].tolist(), [9] * B)  # one PP for both hits

    def test_the_effect_stream_is_read_only_by_a_second_hit(self):
        u = {'moves': [((0, 0, 0), TACKLE)]}
        s = self.build([u, u, {}, {}, {}, {}, {'types': [((1, 0, 0), FLYING)]}, {'types': [((1, 0, 0), FLYING)]}])
        miss = {(0, 6): 99}
        quiet = self.play(s, self.chance(draws=[{}, {}, miss, miss, {}, {}, {}, {}]))
        loud = self.play(s, self.chance(draws=[{}, {}, miss, miss, {}, {}, {}, {}],
                                        effect=[{(0, 0): 0, (0, 1): 0}] * 4 + [{}] * 2 + [{(0, 0): 0}] * 2))
        hp_q, hp_l = np.asarray(quiet.hp)[:, 1, 0], np.asarray(loud.hp)[:, 1, 0]
        self.assertEqual(hp_q[:4].tolist(), hp_l[:4].tolist())             # Tackle; a Bonemerang miss (90%: roll 99)
        self.assertLess(int(hp_q[0]), 300);self.assertEqual(hp_q[2:4].tolist(), [300, 300])
        self.assertEqual(hp_q[6:].tolist(), [300, 300]);self.assertEqual(hp_l[6:].tolist(), [300, 300])   # immune: no hit at all

    def test_substitute_broken_by_the_first_hit_lets_the_second_reach_the_pokemon(self):
        s = self.build([{'substitute_hp': [((1, 0), 10)]}, {'substitute_hp': [((1, 0), 200)]}, {'substitute_hp': [((1, 0), 60)]},
                        {}, {}, {}, {}, {}])
        out = self.play(s)
        hit = self.single_hit(self.build([{}] * B), [15] * B, [15] * B)[0]          # max-roll non-critical hit
        self.assertGreater(int(hit), 10);self.assertLess(int(hit), 60);self.assertLess(int(2 * hit), 200)
        sub, hp = np.asarray(out.substitute_hp)[:, 1, 0], np.asarray(out.hp)[:, 1, 0]
        self.assertEqual((int(sub[0]), int(hp[0])), (0, 300 - hit))                 # broken by hit 1; hit 2 on the mon
        self.assertEqual((int(sub[1]), int(hp[1])), (200 - 2 * hit, 300))          # both hits on the Substitute
        self.assertEqual((int(sub[2]), int(hp[2])), (0, 300))                      # hit 2 breaks it, capped

    def test_a_first_hit_ko_ends_the_move_and_the_block_is_saved(self):
        frail = {'hp': [((1, 0), 20)]}
        s = self.build([frail, frail, {'hp': [((1, 0), 60)]}, {}, {}, {}, {}, {}])
        a = self.chance(effect=[{(0, 0): 0}, {(0, 0): 15}, {(0, 0): 0}])
        b = self.chance(effect=[{(0, 0): 15, (0, 1): 0}, {(0, 0): 0}, {(0, 0): 15, (0, 1): 0}])
        first, second = self.play(s, a), self.play(s, b)
        self.assertEqual(np.asarray(first.hp)[:2, 1, 0].tolist(), [0, 0])
        for name in ('hp', 'substitute_hp', 'boosts', 'status'):
            np.testing.assert_array_equal(np.asarray(getattr(first, name))[:2], np.asarray(getattr(second, name))[:2], name)
        self.assertEqual(np.asarray(first.hp)[2, 1, 0], 0)                          # hit 2 KOs it either way
        # The KO defers the residual: the step's effect block is saved with the others (design 2.1), else zero.
        self.assertEqual(np.asarray(first.pending_residual)[:4].tolist(), [True, True, True, False])
        np.testing.assert_array_equal(np.asarray(first.effect_draws)[:3], np.asarray(a.effect)[:3])
        self.assertFalse(np.asarray(first.effect_draws)[3:].any())

    def test_color_change_after_the_first_hit_retypes_the_second(self):
        # gen4 colorchange onDamagingHit runs after each hit: a Rock Kecleon takes a 2x first hit, turns Ground, and
        # takes the second at 1x; without Color Change both are 2x; a Normal Kecleon takes two neutral hits.
        rock = {'types': [((1, 0, 0), ROCK)]};cc = {'ability': [((1, 0), COLORCHANGE)]}
        s = self.build([self.merge(rock, cc), rock, cc, self.merge(rock, cc, {'substitute_hp': [((1, 0), 200)]}), {}, {}, {}, {}])
        out = self.play(s)
        ref = self.single_hit(self.build([rock, {}] + [{}] * 6), [15] * B, [15] * B)
        strong, plain = int(ref[0]), int(ref[1])
        self.assertGreater(strong, plain)
        taken = 300 - np.asarray(out.hp)[:, 1, 0]
        self.assertEqual(taken[:3].tolist(), [strong + plain, 2 * strong, 2 * plain])
        self.assertEqual(np.asarray(out.types)[:3, 1, 0].tolist(), [[GROUND, 0], [ROCK, 0], [GROUND, 0]])
        self.assertEqual(np.asarray(out.types)[3, 1, 0].tolist(), [ROCK, 0])          # both hits on its Substitute

    def test_endure_caps_each_hit(self):
        endure = {'moves': [((1, 0, 0), ENDURE)]}
        s = self.build([self.merge(endure, {'hp': [((1, 0), 40)]}), self.merge(endure, {'hp': [((1, 0), 60)]}), {}, {}, {}, {}, {}, {}])
        out = self.play(s)
        self.assertEqual(np.asarray(out.hp)[:2, 1, 0].tolist(), [1, 1])

    def test_counter_returns_twice_the_last_hit_that_reached_the_pokemon(self):
        counter = {'moves': [((1, 0, 0), COUNTER)]}
        s = self.build([counter, counter, self.merge(counter, {'substitute_hp': [((1, 0), 10)]}),
                        self.merge(counter, {'substitute_hp': [((1, 0), 200)]}), {}, {}, {}, {}])
        out = self.play(s, self.chance(draws=[{}, {(0, 4): 0}], effect=[{(0, 1): 0}, {(0, 1): 0}]))
        ref = self.build([{}] * B)
        low = int(self.single_hit(ref, [15] * B, [0] * B)[0]);high = int(self.single_hit(ref, [15] * B, [15] * B)[0])
        self.assertNotEqual(low, high)
        given = 300 - np.asarray(out.hp)[:, 0, 0]
        self.assertEqual(given[:4].tolist(), [2 * low, 2 * low, 2 * high, 0])       # the second hit; a Substitute hit counts none

    def test_destiny_bond_on_a_bonemerang_ko(self):
        bond = {'moves': [((1, 0, 0), DESTINYBOND)], 'stats': [((0, 0, 5), 150)]}
        s = self.build([self.merge(bond, {'hp': [((1, 0), 20)]}), self.merge(bond, {'hp': [((1, 0), 60)]}),
                        self.merge(bond, {'hp': [((1, 0), 200)]}), {}, {}, {}, {}, {}])
        out = self.play(s)
        self.assertEqual(np.asarray(out.hp)[:3, 1, 0].tolist()[:2], [0, 0])
        self.assertEqual(np.asarray(out.hp)[:3, 0, 0].tolist(), [0, 0, 300])        # KO on hit 1 or hit 2: both faint

    def test_battle_armor_vetoes_the_second_critical(self):
        armor = {'ability': [((1, 0), BATTLEARMOR)]}
        s = self.build([armor, armor, {}, {}, {}, {}, {}, {}])
        out = self.play(s, self.chance(draws=[{}, {(0, 4): 0}, {}], effect=[{(0, 0): 0}, {(0, 0): 0}, {(0, 0): 0}]))
        hp = np.asarray(out.hp)[:, 1, 0]
        self.assertEqual(int(hp[0]), int(hp[1]));self.assertEqual(int(hp[0]), int(hp[3]))
        self.assertLess(int(hp[2]), int(hp[3]))                                     # without armor the second crit lands

    def test_liquid_ooze_turns_leech_seed_into_damage_to_the_seeder(self):
        # Side 1 is seeded (volatile bit 4); the drain is max_hp//8 = 37 at each residual.
        seeded = {'volatile': [((1, 0), 4)], 'moves': [((0, 0, 0), SPLASH)]}
        ooze = {'ability': [((1, 0), LIQUIDOOZE)]}
        s = self.build([seeded, self.merge(seeded, ooze), self.merge(seeded, ooze, {'hp': [((0, 0), 200)]}),
                        self.merge(seeded, ooze, {'hp': [((0, 0), 20)]}), self.merge(seeded, {'hp': [((0, 0), 200)]}),
                        {'moves': [((0, 0, 0), SPLASH)]}, self.merge(seeded, ooze, {'hp': [((1, 0), 10)]}), seeded])
        out = self.play(s)
        own, foe = np.asarray(out.hp)[:, 0, 0], np.asarray(out.hp)[:, 1, 0]
        self.assertEqual(foe[:6].tolist(), [263, 263, 263, 263, 263, 300])
        self.assertEqual(own[:6].tolist(), [300, 263, 163, 0, 237, 300])          # full HP: no heal; Ooze: damage, a KO too
        self.assertEqual((int(foe[6]), int(own[6])), (0, 290))                    # the drain is capped by the holder's HP
        self.assertTrue(bool(out.forced_switch[3, 0]))
        # Device: the holder's Liquid Ooze is revealed only where it fired.
        from gpu_engine import transition
        init = transition.initialize_with_public(s, self.table)
        actions = self.j.zeros((B, 2), self.j.int32)
        after = self.jax.jit(lambda st, a, tk, c: transition.step_with_public(st, a, self.table, tk, chance_inputs=c))(
            init.state, actions, init.trackers, self.chance())
        self.assertEqual((np.asarray(after.trackers.shown_ability)[:, 1, 0] == LIQUIDOOZE).tolist(),
                         [False, True, True, True, False, False, True, False])

    def test_a_resumed_turn_reads_the_saved_effect_block(self):
        # Side 1 Baton Passed first: side 0's Bonemerang is queued and runs on the resumed replacement step.
        j = self.j
        s = self.build([{}] * B)
        s = s._replace(pending_residual=j.ones(B, bool), queued_actor=j.zeros(B, j.int32), queued_actions=j.zeros((B, 2), j.int32),
            baton_pass=s.baton_pass.at[:, 1].set(True), forced_switch=s.forced_switch.at[:, 1].set(True),
            residual_draws=self.chance().draws)
        saved = np.zeros((B, 2, 4), np.int32);saved[:, :, 0] = 15;saved[:, :, 1] = 15;saved[1::2, 0, 0] = 0   # odd lanes: crit2
        s = s._replace(effect_draws=j.asarray(saved))
        crit_supplied = self.chance(effect=[{(0, 0): 0}] * B)
        quiet_supplied = self.chance()
        a = self.play(s, crit_supplied, a1=4);b = self.play(s, quiet_supplied, a1=4)
        np.testing.assert_array_equal(np.asarray(a.hp), np.asarray(b.hp))
        taken = 300 - np.asarray(a.hp)[:, 1, 1]
        self.assertTrue((taken[1::2] > taken[0::2]).all())                          # the saved crit2, not the supplied one
        self.assertEqual(np.asarray(a.active)[:, 1].tolist(), [1] * B)


class RegressPluginTests(unittest.TestCase):
    """R's plug-in allowances and the A2-review carries L1 (terminal widening) and L2 (trap links)."""
    def test_bonemerang_declares_allowed_fields(self):
        from gpu_engine.v2e_regress import ALLOWED, PLUGINS, lane_verdict
        self.assertTrue(all(e in ALLOWED for e in PLUGINS['R']))
        ok = {('state.hp', 1), ('state.substitute_hp', 1), ("trackers.mon_features['sub_present']", 1),
              ('trackers.public_volatiles', 1), ('state.types', 1), ('trackers.shown_ability', 1), ('state.hp', 0),
              ("trackers.field_features['wish_pending']", 1), ('state.volatile', 1)}
        self.assertEqual(lane_verdict(ok, {'bonemerang': [0]}, None)[0], [])
        for bad in (('state.status', 1), ('state.volatile', 0), ('trackers.public_volatiles', 0), ('trackers.shown_ability', 0),
                    ('state.status', 0), ('state.effect_draws', 0)):
            self.assertEqual(lane_verdict({bad}, {'bonemerang': [0]}, None)[0], [bad])

    def test_a_ko_before_the_targets_turn_cancels_its_bookkeeping(self):
        """A second-hit KO of a slower target cancels its move: its PP and its side's action history (KO side only),
        the executed-move count, and a status that move would have given the user."""
        from gpu_engine.v2e_regress import KO_MON_LEAVES, lane_verdict
        self.assertIn('state.pp', KO_MON_LEAVES)
        effects = {'bonemerang': [1]}
        cascade = dict(ko_sides=np.array([True, False]), mon_ok=lambda leaf, side: side == 0)
        cancelled = {('state.pp', 0), ('state.last_action_kind', 0), ('state.last_action_index', 0),
                     ('step.placeholder_moves', None), ('state.status', 1), ('trackers.public_port.victims', 1)}
        self.assertEqual(lane_verdict(cancelled, effects, None, cascade=cascade)[0], [])
        for bad in (('state.last_action_kind', 1), ('state.pp', 1)):
            self.assertEqual(lane_verdict({bad}, effects, None, cascade=cascade)[0], [bad])
        self.assertEqual(lane_verdict({('step.placeholder_moves', None)}, effects, None)[0], [('step.placeholder_moves', None)])
        # The user's status allowance needs a KO on the target's side (review L1): not without a cascade, and not
        # on a cascade where only the user's own side has a newly fainted mon.
        own_ko = dict(ko_sides=np.array([False, True]), mon_ok=lambda leaf, side: False)
        for verdict in (lane_verdict({('state.status', 1)}, effects, None),
                        lane_verdict({('state.status', 1)}, effects, None, cascade=own_ko)):
            self.assertEqual(verdict[0], [('state.status', 1)])

    def test_trap_links_release_only_for_a_fainted_or_departed_mon(self):
        from gpu_engine.v2e_regress import active_mask, departed_mask, link_ok
        ko = np.zeros((2, 6), bool)
        acting = active_mask(np.array([0, 2])) | active_mask(np.array([0, 2]))
        stay = departed_mask(np.array([0, 2]), np.array([0, 2]), np.array([0, 2]))
        self.assertFalse(stay.any())
        a = np.array([2, -1, -1, -1, -1, -1]);b = np.array([-1, -1, -1, -1, -1, -1])
        # Both actives stay and nobody faints: the active's own link and one pointing at the foe's active no longer pass.
        self.assertFalse(link_ok('state.trap_source', 0, a, b, ko, acting, departed=stay).any())
        # The foe's active (slot 2) left in the base tree only: links pointing at it pass, and only those.
        gone = departed_mask(np.array([0, 2]), np.array([0, 2]), np.array([0, 4]))
        self.assertEqual(gone[1].tolist(), [False, False, True, False, False, False]);self.assertFalse(gone[0].any())
        self.assertEqual(link_ok('state.trap_source', 0, a, b, ko, acting, departed=gone).tolist(), [True] + [False] * 5)
        self.assertEqual(link_ok('state.trap_source', 1, b, b, ko, acting, departed=gone).tolist(),
                         [False, False, True, False, False, False])                  # its own entry
        # A fainted own mon releases its own link; last_move keeps the actives' cancelled actions.
        ko[0, 0] = True
        self.assertEqual(link_ok('state.trap_source', 0, a, b, ko, acting, departed=stay).tolist(), [True] + [False] * 5)
        self.assertEqual(link_ok('state.last_move', 1, None, None, ko, acting, departed=stay).tolist(),
                         [False, False, True, False, False, False])

    def test_terminal_steps_without_a_residual_difference_take_only_end_of_turn_writes(self):
        from gpu_engine.v2e_regress import lane_verdict
        effects = {'bonemerang': [0]}
        cascade = dict(ko_sides=np.array([False, True]), mon_ok=lambda leaf, side: side == 1)
        terminal = dict(cascade, families=('terminal',))
        survivor = {('state.truant_loaf', 0), ("trackers.mon_features['sleep_cant_pending']", 0),
                    ('trackers.public_port.post_upkeep', 0), ('state.done', None), ('state.winner', None)}
        self.assertEqual(lane_verdict(survivor, effects, None, cascade=terminal)[0], [])
        residual = {('state.encore_counter', 0), ('state.weather', None), ('trackers.weather', 0)}
        self.assertEqual(sorted(lane_verdict(residual, effects, None, cascade=terminal)[0]), sorted(residual))
        shifted = dict(cascade, families=('residual', 'residual_trackers'))
        self.assertEqual(lane_verdict(residual, effects, None, cascade=shifted)[0], [])


if __name__ == '__main__':
    unittest.main()
