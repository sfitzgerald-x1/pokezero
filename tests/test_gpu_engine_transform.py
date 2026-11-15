"""Transform (gpu_engine/transform.py and its transition wiring), on authored states.

The rules follow Showdown f76228a1 as cited in gpu_engine/transform.py (data/moves.ts transform, data/mods/gen4/moves.ts
flags, sim/pokemon.ts transformInto/clearVolatile, sim/battle.ts's gen < 5 Update, data/conditions.ts choicelock); the
two-engine probes (gpu_engine/effect_probes/transform_*.json) are the Showdown gate. Light by design, as
test_gpu_engine_fidelity_m2: one jitted step_with_chance at B=8 shared by every transition test (table and arrays are
arguments, compiled once, asserted last), eager calls of the transform helpers, and pure-Python checks of the tables and
future_state.
"""
import unittest

import numpy as np
import jax
import jax.numpy as j

TYPES = ['_none', 'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug',
         'rock', 'ghost', 'dragon', 'dark', 'steel']
NORMAL, FIRE, WATER, FLYING, GHOST = 1, 2, 3, 10, 14
MOVES = ['_none', 'struggle', 'splash', 'transform', 'tackle', 'protect', 'substitute', 'hiddenpower', 'explosion']
STRUGGLE, SPLASH, TRANSFORM, TACKLE, PROTECT, SUB, HP, EXPLOSION = range(1, 9)
ABILITIES = ['_none', 'limber', 'owntempo', 'truant', 'naturalcure', 'pressure', 'insomnia', 'intimidate']
LIMBER, OWNTEMPO, TRUANT, NATURALCURE, PRESSURE, INSOMNIA, INTIMIDATE = range(1, 8)
ITEMS = ['_none', 'choiceband', 'leftovers']
CHOICEBAND = 1
PAR, PSN, SLP = 1, 3, 5
N = 8


def metadata():
    def row(name, power, typ, category, accuracy=100, flags=None, priority=0, pp=10):
        return dict(id=name, base_power=power, type=typ, source_type=TYPES[typ].title(), category=category, pp=pp,
                    accuracy=accuracy, priority=priority, flags={'protect': 1} if flags is None else flags)
    rows = [None, row('struggle', 50, NORMAL, 'Physical', True, {'contact': 1}), row('splash', 0, NORMAL, 'Status', True, {}),
            row('transform', 0, NORMAL, 'Status', True, {'bypasssub': 1, 'metronome': 1, 'failencore': 1}),
            row('tackle', 35, NORMAL, 'Physical', 95, {'contact': 1, 'protect': 1}, pp=35),
            row('protect', 0, NORMAL, 'Status', True, {}, priority=3), row('substitute', 0, NORMAL, 'Status', True, {}),
            row('hiddenpower', 0, NORMAL, 'Special', 100, {'protect': 1}, pp=15),
            row('explosion', 250, NORMAL, 'Physical', 100, {'protect': 1}, pp=5)]
    chart = np.ones((17, 17));chart[NORMAL - 1, GHOST - 1] = 0
    return {'vocabulary': dict(moves=MOVES, abilities=ABILITIES, items=ITEMS, species=['_none', 'a', 'b', 'c', 'd', 'e', 'f'],
                               types=TYPES), 'moves': rows, 'type_chart': chart.tolist()}


class TableTests(unittest.TestCase):
    def test_table_and_leaves(self):
        from gpu_engine import transform
        from gpu_engine.state import BattleState, empty_state
        t = transform.from_metadata(metadata())
        self.assertEqual(np.flatnonzero(np.asarray(t.move)).tolist(), [TRANSFORM])
        self.assertEqual(np.asarray(t.copied_pp).tolist(), [0, 5, 5, 5, 5, 5, 5, 5, 5])
        names = ('transformed', 'transform_species', 'transform_stats', 'transform_moves', 'transform_pp', 'transform_hidden_power')
        start = BattleState._fields.index('transformed')
        self.assertEqual(BattleState._fields[start - 1:start + 6], ('partial_trap_source',) + names)
        s = empty_state(3)
        self.assertEqual([tuple(getattr(s, n).shape) for n in names], [(3, 2), (3, 2), (3, 2, 5), (3, 2, 4), (3, 2, 4), (3, 2, 2)])
        self.assertFalse(bool(np.asarray(s.transformed).any()))

    def test_future_state_policy_and_refusal(self):
        from gpu_engine import future_state as f
        for name in f.TRANSFORM_FIELDS:
            self.assertIn(name, f.FIELD_POLICY);self.assertIn(name, f.NATIVE_PATHS);self.assertIn(name, f.SHAPES)
        with open(f.__file__) as source:self.assertIn('TRANSFORM_UNREPRESENTED', source.read())


class HelperTests(unittest.TestCase):
    """transform.apply/revert/update_cure/choice_unlock called eagerly on a two-lane authored state."""
    def state(self):
        from gpu_engine.state import empty_state
        s = empty_state(2)
        stats = np.full((2, 2, 6, 6), 100, np.int32);stats[:, 1, 0] = [300, 210, 220, 230, 240, 250]
        moves = np.zeros((2, 2, 6, 4), np.int32);moves[:, 0, 0, 0] = TRANSFORM;moves[:, 1, 0] = [TACKLE, PROTECT, HP, 0]
        pp = np.zeros((2, 2, 6, 4), np.int32);pp[:, 0, 0, 0] = 15;pp[:, 1, 0] = [56, 16, 24, 0]
        types = np.zeros((2, 2, 6, 2), np.int32);types[..., 0] = NORMAL;types[:, 1, 0] = [FIRE, FLYING]
        boosts = np.zeros((2, 2, 6, 7), np.int32);boosts[:, 1, 0] = [2, 0, 0, 0, -1, 0, 1]
        ability = np.full((2, 2, 6), LIMBER, np.int32);ability[:, 1, 0] = OWNTEMPO
        species = np.broadcast_to(np.arange(1, 7, dtype=np.int32), (2, 2, 6)).copy()
        hpt = np.zeros((2, 2, 6), np.int32);hpt[:, 1, 0] = WATER
        hpp = np.full((2, 2, 6), 70, np.int32);hpp[:, 1, 0] = 60
        return s._replace(hp=j.full((2, 2, 6), 100), max_hp=j.full((2, 2, 6), 100), stats=j.asarray(stats), moves=j.asarray(moves),
            pp=j.asarray(pp), types=j.asarray(types), base_types=j.asarray(types), boosts=j.asarray(boosts),
            ability=j.asarray(ability), base_ability=j.asarray(ability), species=j.asarray(species),
            hidden_power_type=j.asarray(hpt), hidden_power_power=j.asarray(hpp), done=j.zeros(2, bool),
            last_action_kind=j.full((2, 2), 2, j.int32), last_action_index=j.zeros((2, 2), j.int32))

    def test_apply_copies_saves_and_revert_restores(self):
        from gpu_engine import transform
        table = transform.from_metadata(metadata())
        s = self.state()
        before = {k: np.asarray(v) for k, v in s._asdict().items()}
        out, copied = transform.apply(s, 0, j.asarray([True, False]), table)
        self.assertEqual(np.asarray(copied).tolist(), [True, False])
        o = {k: np.asarray(v) for k, v in out._asdict().items()}
        self.assertEqual(o['stats'][0, 0, 0].tolist(), [100, 210, 220, 230, 240, 250])       # HP stays
        self.assertEqual(o['types'][0, 0, 0].tolist(), [FIRE, FLYING])
        self.assertEqual(o['moves'][0, 0, 0].tolist(), [TACKLE, PROTECT, HP, 0])
        self.assertEqual(o['pp'][0, 0, 0].tolist(), [5, 5, 5, 0])
        self.assertEqual(o['boosts'][0, 0, 0].tolist(), [2, 0, 0, 0, -1, 0, 1])
        self.assertEqual((int(o['ability'][0, 0, 0]), int(o['base_ability'][0, 0, 0])), (OWNTEMPO, LIMBER))
        self.assertEqual((int(o['hidden_power_type'][0, 0, 0]), int(o['hidden_power_power'][0, 0, 0])), (WATER, 60))
        self.assertEqual((bool(o['transformed'][0, 0]), int(o['transform_species'][0, 0]), int(o['species'][0, 0, 0])), (True, 1, 1))
        self.assertEqual(o['transform_stats'][0, 0].tolist(), [100] * 5)
        self.assertEqual(o['transform_moves'][0, 0].tolist(), [TRANSFORM, 0, 0, 0])
        self.assertEqual(o['transform_pp'][0, 0].tolist(), [15, 0, 0, 0])
        self.assertEqual(o['transform_hidden_power'][0, 0].tolist(), [0, 70])
        self.assertEqual(o['last_action_index'][:, 0].tolist(), [-1, 0])
        for k, v in before.items():          # lane 1 (not eligible) and the foe are untouched
            if k == 'keys':continue
            np.testing.assert_array_equal(o[k][1], v[1], err_msg=k)
            if v.ndim > 1 and v.shape[1] == 2:np.testing.assert_array_equal(o[k][:, 1], v[:, 1], err_msg=k)
        # A re-transform keeps the true base; the foe already transformed fails (only the history slot moves).
        again, copied = transform.apply(out, 0, j.asarray([True, True]), table)
        self.assertEqual(np.asarray(again.transform_moves)[0, 0].tolist(), [TRANSFORM, 0, 0, 0])
        blocked = out._replace(transformed=out.transformed.at[:, 1].set(True))
        failed, copied = transform.apply(blocked, 0, j.asarray([False, True]), table)
        self.assertEqual(np.asarray(copied).tolist(), [False, False])
        self.assertEqual(np.asarray(failed.moves)[1, 0, 0].tolist(), [TRANSFORM, 0, 0, 0])
        self.assertEqual(np.asarray(failed.last_action_index)[1, 0], -1)
        back, reverted = transform.revert(out, 0, True)
        self.assertEqual(np.asarray(reverted).tolist(), [True, False])
        b = {k: np.asarray(v) for k, v in back._asdict().items()}
        for k in ('stats', 'types', 'moves', 'pp', 'ability', 'hidden_power_type', 'hidden_power_power', 'transformed',
                  'transform_species', 'transform_stats', 'transform_moves', 'transform_pp', 'transform_hidden_power'):
            np.testing.assert_array_equal(b[k], before[k], err_msg=k)
        self.assertEqual(b['boosts'][0, 0, 0].tolist(), [2, 0, 0, 0, -1, 0, 1])          # stages are the caller's

    def test_update_cure_and_choice_unlock(self):
        from gpu_engine import transform, entry
        table = transform.from_metadata(metadata());entries = entry.from_metadata(metadata())
        s = self.state()
        s = s._replace(status=s.status.at[:, 0, 0].set(PAR), confusion_counter=s.confusion_counter.at[:, 0, 0].set(3),
                       ability=s.ability.at[0, 1, 0].set(LIMBER))
        out, copied = transform.apply(s, 0, True, table)
        cured, by_status, by_tempo = transform.update_cure(out, 0, copied, entries)
        self.assertEqual(np.asarray(by_status).tolist(), [True, False]);self.assertEqual(np.asarray(by_tempo).tolist(), [False, True])
        self.assertEqual(np.asarray(cured.status)[:, 0, 0].tolist(), [0, PAR])
        self.assertEqual(np.asarray(cured.confusion_counter)[:, 0, 0].tolist(), [3, 0])
        locked = out._replace(choice_move=out.choice_move.at[:, 0, 0].set(j.asarray([TRANSFORM, TACKLE])))
        unlocked = transform.choice_unlock(locked, True)
        self.assertEqual(np.asarray(unlocked.choice_move)[:, 0, 0].tolist(), [0, TACKLE])
        plain = s._replace(choice_move=s.choice_move.at[:, 0, 0].set(TACKLE))            # not transformed: untouched
        self.assertEqual(np.asarray(transform.choice_unlock(plain, True).choice_move)[:, 0, 0].tolist(), [TACKLE, TACKLE])


class ConstructorTests(unittest.TestCase):
    """The constructor helpers in this PR's files (gpu_engine/phase_d_constructor_rules, constructor_worlds,
    device_public_world_state); Codex's materializers that consume them are a separate patch (no engine step here)."""
    def test_own_pp_carry_returns_the_base_ledger_of_a_transformed_active(self):
        """The walls fixture's Ditto (one move, Transform 16 PP) at its first transformed request: the copied PP stays the
        request's, and transform_base gives the -transform line's ledger (Transform paid: 15)."""
        import json
        from pathlib import Path
        from pokezero.live_protocol_actor import _packet
        from gpu_engine.phase_d_constructor_rules import own_pp_carry
        data = json.loads((Path(__file__).parent / 'fixtures/phase_d_constructor_walls/ditto_transform.json').read_text())
        lines = data['request']['lines'];owner = data['request']['owner']
        at = next(i for i, line in enumerate(lines) if line.startswith('|request|') and '"Hidden Power Grass 70"' in line)
        packet = _packet(lines[at], owner);maximum = data['own_pp_carry']['maximum_pp']
        rows, base = own_pp_carry(lines[:at + 1], packet, maximum, transform_base=True)
        self.assertEqual(rows, own_pp_carry(lines[:at + 1], packet, maximum))
        ditto = next(i for i, row in enumerate(packet['side']['pokemon']) if row['ident'].endswith(': Ditto'))
        self.assertEqual(base[ditto], [('transform', 15)])
        self.assertEqual([b for k, b in enumerate(base) if k != ditto], [None] * 5)

    def test_line_facts_and_the_copied_pp_fills(self):
        """line_facts' TransformFacts: the donor and the copied uses since the latest -transform (Pressure doubles one known to
        the owner); a switch ends it. fix_copied_pp / fill_copied_pp write the peer's copied PP and the carry's uses only on a
        world whose state already holds the peer's Transform."""
        from gpu_engine import constructor_worlds as cw
        from gpu_engine.state import empty_state
        from gpu_engine import engine_device_trackers as tr
        lines = ['|switch|p1a: Ditto|Ditto|258/258', '|switch|p2a: Zapdos|Zapdos, L74|100/100', '|turn|1',
                 '|-ability|p2a: Zapdos|Pressure', '|move|p1a: Ditto|Transform|p2a: Zapdos', '|-transform|p1a: Ditto|p2a: Zapdos',
                 '|turn|2', '|move|p1a: Ditto|Thunderbolt|p2a: Zapdos', '|turn|3', '|move|p1a: Ditto|Thunderbolt|p2a: Zapdos',
                 '|move|p1a: Ditto|Sleep Talk|p1a: Ditto', '|move|p1a: Ditto|Protect|p1a: Ditto|[from] move: Sleep Talk']
        facts = cw.line_facts(lines, 1)
        self.assertEqual(facts.transform, (cw.TransformFacts('zapdos', (('thunderbolt', 4), ('sleeptalk', 1))), None))
        self.assertEqual(cw.line_facts(lines + ['|switch|p1a: Mew|Mew, L72|100/100'], 1).transform, (None, None))
        names = ['_none', 'thunderbolt', 'sleeptalk', 'protect']
        tables = cw._table(np.zeros((3, 2), np.int32), move_names=names, foe_targeted=[False] * 4)
        s = empty_state(1)._replace(moves=j.zeros((1, 2, 6, 4), j.int32).at[0, 0, 0].set(j.asarray([1, 2, 3, 0])),
                                    pp=j.full((1, 2, 6, 4), 9, j.int32))
        self.assertIs(cw.fix_copied_pp(s, 1, facts, tables), s)                     # the world does not hold the Transform
        held = s._replace(transformed=s.transformed.at[0, 0].set(True))
        self.assertEqual(np.asarray(cw.fix_copied_pp(held, 1, facts, tables).pp)[0, 0, 0].tolist(), [1, 4, 5, 9])
        public = tr.Trackers(*([j.zeros((1,), j.int32)] * 17), copied_pp_used=j.zeros((1, 2, 4), j.int32))
        self.assertEqual(np.asarray(cw.fill_copied_pp(held, public, 1, facts, tables).copied_pp_used)[0, 0].tolist(), [4, 1, 0, 0])

    def test_device_world_rebuilds_both_copies_from_the_world(self):
        """_materialize_transform: our transformed active's hidden copied stats and ability come from the world's sampled donor
        (not the actual root), and the opponent's transformed active becomes a copy of our donor over its sampled base."""
        from types import SimpleNamespace as NS
        from gpu_engine import device_public_world_state as dws
        from gpu_engine.state import empty_state
        arrays = {k: np.array(v, copy=True) for k, v in empty_state(1)._asdict().items()}
        arrays['species'][0] = [[11, 12, 13, 14, 15, 16], [21, 22, 23, 24, 25, 26]]
        arrays['stats'][0, 1, 0] = [300, 201, 202, 203, 204, 205];arrays['stats'][0, 0, 1] = [250, 101, 102, 103, 104, 105]
        arrays['stats'][0, 0, 0] = [258, 999, 999, 999, 999, 999]      # the TRUE donor's, as the actual root copied them
        arrays['ability'][0, 1, 0] = 7;arrays['ability'][0, 0, 0] = 9;arrays['base_ability'][0, 0, 1] = 4
        arrays['moves'][0, 0, 1] = [1, 2, 0, 0];arrays['moves'][0, 1, 1] = [5, 6, 0, 0];arrays['pp'][0, 1, 1] = [10, 3, 0, 0]
        arrays['base_types'][0, 0, 1] = [3, 0]
        tables = NS(namespaces=dict(abilities={'trace': 50, 'colorchange': 51}, items={}), copied_pp=tuple([0] + [5] * 9))
        own = dict(transformed=np.bool_(True), transform_species=np.int32(21), active=np.int32(0),
                   transform_stats=np.zeros(5, np.int32), transform_moves=np.zeros(4, np.int32), transform_hidden_power=np.zeros(2, np.int32))
        view = dict(transformed=np.zeros((1, 6), bool), transform_species=np.zeros((1, 6), np.int32))
        root = NS(own=own, view=view, public=NS(copied_pp_used=None))
        dws._materialize_transform(root, tables, arrays, 0, 1, 1, {0: 0, 1: 1})
        self.assertEqual(arrays['stats'][0, 0, 0].tolist(), [258, 201, 202, 203, 204, 205])
        self.assertEqual(int(arrays['ability'][0, 0, 0]), 7)
        own['transformed'] = np.bool_(False)
        view['transformed'][0, 1] = True;view['transform_species'][0, 1] = 12
        root = NS(own=own, view=view, public=NS(copied_pp_used=j.asarray([[[0, 0, 0, 0], [2, 0, 0, 0]]], j.int32)))
        dws._materialize_transform(root, tables, arrays, 0, 1, 1, {0: 0, 1: 1})
        self.assertEqual(arrays['stats'][0, 1, 1, 1:].tolist(), [101, 102, 103, 104, 105])
        self.assertEqual(arrays['moves'][0, 1, 1].tolist(), [1, 2, 0, 0]);self.assertEqual(arrays['pp'][0, 1, 1].tolist(), [3, 5, 0, 0])
        self.assertEqual(arrays['transform_moves'][0, 1].tolist(), [5, 6, 0, 0]);self.assertEqual(arrays['transform_pp'][0, 1].tolist(), [10, 3, 0, 0])
        self.assertEqual((int(arrays['ability'][0, 1, 1]), arrays['types'][0, 1, 1].tolist()), (4, [3, 0]))
        self.assertEqual((bool(arrays['transformed'][0, 1]), int(arrays['transform_species'][0, 1])), (True, 12))
        arrays['base_ability'][0, 0, 1] = 50
        with self.assertRaisesRegex(ValueError, 'PUBLIC_TRANSFORM_DONOR_OVERRIDDEN'):
            dws._materialize_transform(root, tables, arrays, 0, 1, 1, {0: 0, 1: 1})


class TransitionTests(unittest.TestCase):
    """step_with_chance on authored 8-lane states: side 0 (faster) uses its slot-0 move, side 1 its slot 0 (Splash by
    default). One jit, the table and every array as arguments, compiled once (asserted last)."""
    @classmethod
    def setUpClass(cls):
        from gpu_engine import transition
        from gpu_engine.state import BattleState, empty_state
        cls.t, cls.State, cls.empty = transition, BattleState, staticmethod(empty_state)
        cls.table = transition.tables_from_metadata(metadata())
        cls.jitted = dict(step=jax.jit(transition.step_with_chance))

    def build(self, lanes, move=TRANSFORM, foe_move=SPLASH):
        s = self.empty(N, seed=4246)
        stats = np.full((N, 2, 6, 6), 200, np.int32);stats[..., 0] = 320;stats[:, 0, :, 5] = 301
        stats[:, 1, 0] = [320, 250, 180, 260, 170, 210]
        types = np.zeros((N, 2, 6, 2), np.int32);types[..., 0] = NORMAL;types[:, 1, 0] = [FIRE, FLYING]
        moves = np.full((N, 2, 6, 4), SPLASH, np.int32);moves[:, 0, :, 0] = move;moves[:, 1, :, 0] = foe_move
        moves[:, 1, 0, 1:] = [TACKLE, HP, 0]
        ability = np.full((N, 2, 6), LIMBER, np.int32);ability[:, 1, 0] = INTIMIDATE
        arrays = {k: np.array(v) for k, v in s._asdict().items()}
        arrays.update(hp=np.full((N, 2, 6), 320, np.int32), max_hp=np.full((N, 2, 6), 320, np.int32), stats=stats,
                      level=np.full((N, 2, 6), 100, np.int32), types=types, moves=moves, pp=np.full((N, 2, 6, 4), 10, np.int32),
                      ability=ability, species=np.broadcast_to(np.arange(1, 7, dtype=np.int32), (N, 2, 6)).copy(),
                      done=np.zeros(N, bool), last_action_kind=np.full((N, 2), 2, np.int32),
                      last_action_index=np.zeros((N, 2), np.int32))
        assert len(lanes) <= N
        for lane, edits in enumerate(lanes):
            for field, writes in edits.items():
                for index, value in writes:arrays[field][(lane,) + tuple(index)] = value
        arrays['base_types'] = arrays['types'].copy();arrays['base_ability'] = arrays['ability'].copy()
        for lane, edits in enumerate(lanes):          # authored copies keep their own base
            for field in ('base_types', 'base_ability'):
                for index, value in edits.get(field, ()):arrays[field][(lane,) + tuple(index)] = value
        return self.State(**{k: j.asarray(v, dtype=getattr(s, k).dtype) for k, v in arrays.items()})

    def chance(self, draws=()):
        """Hit, no crit, max damage roll, no thaw/paralysis/self-hit, Protect succeeds."""
        d = np.zeros((N, 2, 12), np.int32);d[:, :, 1:3] = 99;d[:, :, 4] = 15;d[:, :, 5] = 15
        for lane, edits in enumerate(draws):
            for (side, column), value in edits.items():d[lane, side, column] = value
        z = j.zeros((N, 2), j.int32);e = np.zeros((N, 2, 4), np.int32);e[:, :, 0] = 15
        return self.t.ChanceInputs(draws=j.asarray(d), phazing=z, secondary=j.full((N, 2), 99, j.int32), sleep_talk=z, entry_ties=z,
                                   initial_tie=j.zeros(N, j.int32), effect=j.asarray(e))

    def play(self, s, chance, a0=0, a1=0):
        actions = j.zeros((N, 2), j.int32).at[:, 0].set(j.asarray(a0, j.int32)).at[:, 1].set(j.asarray(a1, j.int32))
        out = self.jitted['step'](s, actions, self.table, chance)
        self.assertFalse(bool(np.asarray(out.invalid_actions).any()), np.asarray(out.invalid_actions))
        return out.state

    def test_transform_copies_through_protect_substitute_and_immunity(self):
        """Lanes: 0 plain; 1 the foe is already transformed (-fail, PP spent); 2 the foe Protects first; 3 the foe holds a
        Substitute; 4 the foe's Pressure (2 PP); 5 a Ghost foe (a Status move ignores immunity); 6 a user asleep (no move);
        7 a re-transform: the user already copies a Mew that knows Transform, so its saves stay."""
        lanes = [{}, {'transformed': [((1,), True)], 'transform_species': [((1,), 3)]},
                 {'moves': [((1, 0, 0), PROTECT)]}, {'substitute_hp': [((1, 0), 80)]}, {'ability': [((1, 0), PRESSURE)]},
                 {'types': [((1, 0, 0), GHOST)]}, {'status': [((0, 0), SLP)], 'status_counter': [((0, 0), 3)]},
                 {'transformed': [((0,), True)], 'transform_species': [((0,), 5)], 'transform_moves': [((0, 0), SPLASH)],
                  'transform_pp': [((0, 0), 7)], 'transform_stats': [((0, k), 111) for k in range(5)],
                  'transform_hidden_power': [((0, 1), 70)], 'pp': [((0, 0, 0), 5)], 'ability': [((0, 0), OWNTEMPO)],
                  'base_ability': [((0, 0), LIMBER)]}]
        s = self.play(self.build(lanes), self.chance())
        tr = np.asarray(s.transformed)[:, 0].tolist()
        self.assertEqual(tr, [True, False, True, True, True, True, False, True])
        moves = np.asarray(s.moves)[:, 0, 0];pp = np.asarray(s.pp)[:, 0, 0]
        for lane in (0, 2, 3, 4, 5, 7):
            self.assertEqual(moves[lane].tolist(), [SPLASH if lane != 2 else PROTECT, TACKLE, HP, 0], lane)
            self.assertEqual(pp[lane].tolist(), [5, 5, 5, 0], lane)
            self.assertEqual(np.asarray(s.stats)[lane, 0, 0].tolist(), [320, 250, 180, 260, 170, 210], lane)
            self.assertEqual(np.asarray(s.ability)[lane, 0, 0], PRESSURE if lane == 4 else INTIMIDATE, lane)
            self.assertEqual(np.asarray(s.transform_species)[lane, 0], 1, lane)
        self.assertEqual(np.asarray(s.types)[0, 0, 0].tolist(), [FIRE, FLYING]);self.assertEqual(np.asarray(s.types)[5, 0, 0].tolist(), [GHOST, FLYING])
        self.assertEqual(np.asarray(s.transform_pp)[:, 0, 0].tolist()[:6], [9, 0, 9, 9, 8, 9])      # paid before the save
        self.assertEqual(np.asarray(s.pp)[1, 0, 0].tolist(), [9, 10, 10, 10])                       # -fail: PP spent, no copy
        self.assertEqual(np.asarray(s.transform_moves)[7, 0].tolist(), [SPLASH, 0, 0, 0])          # re-transform keeps the base
        self.assertEqual(np.asarray(s.transform_stats)[7, 0].tolist(), [111] * 5)
        self.assertEqual(np.asarray(s.last_action_index)[:, 0].tolist(), [-1, -1, -1, -1, -1, -1, 0, -1])
        self.assertEqual(np.asarray(s.last_move)[[0, 1], 0, 0].tolist(), [TRANSFORM, TRANSFORM])
        # A copied Intimidate does nothing (setAbility runs Start only for gen > 3): the foe's Attack stage stays 0.
        self.assertEqual(np.asarray(s.boosts)[0, 1, 0, 0], 0)

    def test_revert_at_switch_faint_natural_cure_and_truant_clock(self):
        """Already-transformed side-0 actives. Lane 0 switches out: the base returns (stats, moves, PP as last paid, Hidden
        Power, ability, types), flag and saves clear. Lane 1 faints to the foe's Tackle at 1 HP: the faint revert.
        Lane 2 a copied Natural Cure cures at the switch. Lane 3 a copied Truant's clock stays with the
        departing mon; lane 4 a real Truant holder's clock resets as before."""
        copy = {'transformed': [((0,), True)], 'transform_species': [((0,), 2)], 'transform_moves': [((0, 0), TRANSFORM)],
                'transform_pp': [((0, 0), 15)], 'transform_stats': [((0, k), 150 + k) for k in range(5)],
                'transform_hidden_power': [((0, 0), WATER), ((0, 1), 70)], 'moves': [((0, 0, 0), TACKLE)],
                'pp': [((0, 0, 0), 5)], 'hidden_power_type': [((0, 0), FIRE)], 'hidden_power_power': [((0, 0), 60)],
                'types': [((0, 0, 0), FIRE)], 'base_types': [((0, 0, 0), NORMAL)], 'boosts': [((0, 0, 0), 2)]}
        def with_(**edits):
            out = {k: list(v) for k, v in copy.items()}
            for k, v in edits.items():out[k] = out.get(k, []) + v
            return out
        lanes = [with_(ability=[((0, 0), INTIMIDATE)], base_ability=[((0, 0), LIMBER)]),
                 with_(hp=[((0, 0), 1)], ability=[((0, 0), INTIMIDATE)], base_ability=[((0, 0), LIMBER)]),
                 with_(ability=[((0, 0), NATURALCURE)], base_ability=[((0, 0), LIMBER)], status=[((0, 0), PSN)]),
                 with_(ability=[((0, 0), TRUANT)], base_ability=[((0, 0), LIMBER)], truant_loaf=[((0, 0), True)]),
                 {'ability': [((0, 0), TRUANT)], 'truant_loaf': [((0, 0), True)]}]
        s = self.play(self.build(lanes, foe_move=TACKLE), self.chance(), a0=j.asarray([4, 0, 4, 4, 4, 0, 0, 0]))
        o = {k: np.asarray(v) for k, v in s._asdict().items()}
        for lane in (0, 1, 2, 3):
            self.assertFalse(bool(o['transformed'][lane, 0]), lane)
            self.assertEqual(o['stats'][lane, 0, 0, 1:].tolist(), [150, 151, 152, 153, 154], lane)
            self.assertEqual(o['moves'][lane, 0, 0].tolist(), [TRANSFORM, 0, 0, 0], lane)
            self.assertEqual(o['pp'][lane, 0, 0].tolist(), [15, 0, 0, 0], lane)
            self.assertEqual((o['hidden_power_type'][lane, 0, 0], o['hidden_power_power'][lane, 0, 0]), (WATER, 70), lane)
            self.assertEqual((o['ability'][lane, 0, 0], o['types'][lane, 0, 0, 0]), (LIMBER, NORMAL), lane)
            self.assertEqual(o['boosts'][lane, 0, 0].tolist(), [0] * 7, lane)
            self.assertEqual((o['transform_species'][lane, 0], o['transform_moves'][lane, 0].sum()), (0, 0), lane)
        self.assertEqual(o['hp'][1, 0, 0], 0)
        self.assertEqual(o['status'][2, 0, 0], 0)
        self.assertEqual(o['truant_loaf'][[3, 4], 0, 0].tolist(), [True, False])

    def test_update_cure_choice_lock_and_copied_truant(self):
        """Lane 0: a paralyzed user copies Limber: cured right after its action. Lane 1: a confused user copies Own Tempo.
        Lane 2: a Choice Band user locks on Transform, then holds the copied moves: the end-of-turn DisableMove drops it.
        Lane 3: the donor knows Transform: the lock stays. Lane 4: a copied Truant flips at its residual."""
        lanes = [{'status': [((0, 0), PAR)], 'ability': [((1, 0), LIMBER)]},
                 {'confusion_counter': [((0, 0), 3)], 'ability': [((1, 0), OWNTEMPO)]},
                 {'item': [((0, 0), CHOICEBAND)]},
                 {'item': [((0, 0), CHOICEBAND)], 'moves': [((1, 0, 2), TRANSFORM)]},
                 {'ability': [((1, 0), TRUANT)]}]
        s = self.play(self.build(lanes), self.chance())
        o = {k: np.asarray(v) for k, v in s._asdict().items()}
        self.assertEqual(o['status'][0, 0, 0], 0)
        self.assertEqual(o['confusion_counter'][1, 0, 0], 0)
        self.assertEqual(o['choice_move'][[2, 3], 0, 0].tolist(), [0, TRANSFORM])
        self.assertEqual((o['ability'][4, 0, 0], bool(o['truant_loaf'][4, 0, 0])), (TRUANT, True))

    def test_zz_one_compile(self):
        """Every transition test used the one compiled step (no recompile)."""
        self.assertEqual(self.jitted['step']._cache_size(), 1)


if __name__ == '__main__':
    unittest.main()
