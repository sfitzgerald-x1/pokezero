"""scripts/v2_offline_view_parity.py: pure pieces on synthetic arrays, the true-world builder on a real tracker carry and
real world-building code, and one CPU end-to-end smoke.

The smoke's fixture is synthetic but its world machinery is real: a seven-species source with two variants each, the real
encoder tables compiled from it (engine_device_observation.compile_tables over a v4 catalog), the real public-hook
tracker (engine_device_public_hooks.initialize/install), the offline path's own make_step, detach_roots, plan_starts,
materialize_host, _rebase_host and the FramePricer's views. Only the engine step and the actor are stand-ins: a scripted
step (switches, fixed damage, faints; trackers through the real hooks) and a forward whose logits follow a script
keyed by the public turn. On that consistent game every peer cell must match after the named redactions and the
bench-order alignment, every owner world must be exact, and the scripted back-switch must show as an order-only
difference.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from scripts import v2_offline_view_parity as parity
from _showdown_root import requires_showdown

SHOWDOWN = requires_showdown('the synthetic fixture compiles its catalog from a pokemon-showdown checkout')


# ---------------------------------------------------------------------------------------------- synthetic world fixture

SPECIES = ('Snorlax', 'Gengar', 'Starmie', 'Skarmory', 'Blissey', 'Tyranitar', 'Jolteon')
MOVES = ('bodyslam', 'surf', 'rest', 'protect', 'toxic', 'psychic')
VARIANT_MOVES = (('protect', 'rest', 'surf', 'bodyslam'), ('bodyslam', 'surf', 'rest', 'protect'))
_FIXTURE = {}


def fixture():
    """(raw source, metadata, catalog, observation tables, compiled plans.Source, materializer tables), built once."""
    if _FIXTURE:return SimpleNamespace(**_FIXTURE)
    from pokezero import v5_contract
    from pokezero.category_vocab import CategoryVocabulary
    from pokezero.dex import ShowdownDex, load_showdown_dex_cached, normalize_id
    from pokezero.randbat import Gen3RandbatSource, Gen3RandbatSpeciesUniverse, Gen3RandbatVariant, RandbatSourceMetadata
    from gpu_engine import v4_encoder, public_belief, engine_device_observation as encoder
    from gpu_engine import device_public_worlds as worlds, device_public_world_state as states
    import test_gpu_engine_v4_encoder as v4
    from _showdown_root import showdown_root
    v4.V4EncoderTests.setUpClass();base = v4.V4EncoderTests.catalog
    dex = load_showdown_dex_cached(showdown_root())
    universes = {normalize_id(name): Gen3RandbatSpeciesUniverse(name, 80, tuple(Gen3RandbatVariant('%s-%d' % (name, v), 'authored', name,
        'authored', 80, VARIANT_MOVES[v], ('Insomnia', 'Pressure')[v], ('Chesto Berry', 'Leftovers')[v]) for v in range(2))) for name in SPECIES}
    species = {normalize_id(n): dex.species_info(normalize_id(n)) for n in SPECIES}
    source = Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle', 3, None, None, None, 'synthetic-parity-source'), universes,
        move_metadata={m: {'type': dex.move_info(m).type, 'category': dex.move_info(m).category} for m in MOVES},
        species_metadata={k: dict(baseStats=dict(v.base_stats), types=list(v.types), gender='N') for k, v in species.items()})
    moves = {m: dex.move_info(m) for m in MOVES + ('struggle',)}
    labels = set(base.vocabulary.tokens) | {'species:' + k for k in species}
    for m in moves:labels |= {'move:' + m, 'belief:possible_move:' + m}
    for a in ('pressure', 'insomnia'):labels |= {'ability:' + a, 'belief:possible_ability:' + a}
    for i in ('leftovers', 'chestoberry'):labels |= {'item:' + i, 'belief:possible_item:' + i}
    catalog = v4_encoder.compile_encoder(public_belief.compile_source(source), ShowdownDex(moves, species, dex.type_chart),
        CategoryVocabulary(tuple(sorted(labels))), checkpoint_sha256='a' * 64, feature_masks=base.masks.__dict__)
    types = ['_none'] + sorted({t.lower() for v in species.values() for t in v.types})
    metadata = dict(vocabulary=dict(species=['_none', *species], types=types, abilities=['_none', 'insomnia', 'pressure'],
        items=['_none', 'chestoberry', 'leftovers'], moves=['_none', *MOVES, 'struggle']),
        moves=[None] + [dict(pp=dex.move_info(m).pp) for m in MOVES + ('struggle',)])
    tables = encoder.compile_tables(metadata, catalog=catalog)
    compiled = worlds.compile_source(source, metadata, expected_source_hash='synthetic-parity-source',
        expected_source_rows_sha256=v5_contract.source_variant_census(source)['source_rows_sha256'])
    _FIXTURE.update(source=source, metadata=metadata, catalog=catalog, tables=tables, compiled=compiled,
        materializer=states.compile_tables(compiled, metadata))
    return SimpleNamespace(**_FIXTURE)


def member(f, name, variant, permutation=(0, 1, 2, 3)):
    """Engine cells of one real team member built exactly as the materializer builds a source fixture."""
    from pokezero.dex import normalize_id
    from pokezero.determinization import _fixture_from_variant_payload
    from pokezero.gen3_damage import gen3_hp_stat, gen3_stat
    v = f.source.universe_for(name).variants[variant];vocab = f.metadata['vocabulary']
    mon = _fixture_from_variant_payload(v.to_summary(), fallback_species=name, set_source=f.source, gender='N')
    info = f.source.species_metadata[normalize_id(name)];base = info['baseStats']
    stats = [gen3_hp_stat(base['hp'], mon.ivs['hp'], mon.evs['hp'], 80)] + [gen3_stat(base[k], mon.ivs[k], mon.evs[k], 80)
        for k in ('atk', 'def', 'spa', 'spd', 'spe')]
    moves = [vocab['moves'].index(normalize_id(v.moves[k])) for k in permutation]
    kinds = [vocab['types'].index(t.lower()) for t in info['types']] + [0] * (2 - len(info['types']))
    pp = [f.metadata['moves'][m]['pp'] * 8 // 5 for m in moves]
    return dict(species=vocab['species'].index(normalize_id(name)), level=80, stats=stats, hp=stats[0], types=kinds, moves=moves, pp=pp,
        ability=vocab['abilities'].index(normalize_id(v.ability)), item=vocab['items'].index(normalize_id(v.item)), variant=variant)


def teams_state(f, teams):
    """A fresh BattleState and constructor truth for teams[row][side] = six member() dicts."""
    from gpu_engine.state import empty_state
    from gpu_engine.v5_device_rollout import ConstructorTruth
    b = len(teams);s = empty_state(b, seed=5)
    cells = {k: np.array(getattr(s, k), copy=True) for k in s._fields}
    probabilities = np.zeros((b, 2, 6, 64), np.float32)
    for r, row in enumerate(teams):
        for side, team in enumerate(row):
            for p, m in enumerate(team):
                for name in ('species', 'level', 'hp', 'ability', 'item'):cells[name][r, side, p] = m[name]
                cells['base_ability'][r, side, p] = m['ability'];cells['max_hp'][r, side, p] = m['stats'][0]
                cells['stats'][r, side, p] = m['stats'];cells['types'][r, side, p] = cells['base_types'][r, side, p] = m['types']
                cells['moves'][r, side, p] = m['moves'];cells['pp'][r, side, p] = m['pp'];cells['gender'][r, side, p] = 0
                probabilities[r, side, p, m['variant']] = 1
    cells.update(active=np.zeros((b, 2), np.int32), done=np.zeros(b, bool), entry_pending=np.zeros((b, 2), bool))
    state = type(s)(**{k: j.asarray(v) for k, v in cells.items()})
    truth = ConstructorTruth(j.asarray(probabilities), j.zeros((b, 2, 6), j.int32), j.ones((b, 2, 6), bool))
    return state, truth


def team(f, names, variants, permutation=(0, 1, 2, 3)):
    return [member(f, n, v, permutation) for n, v in zip(names, variants)]


def scripted_step(state, actions, table, trackers, *, belief_tables=None, chance_inputs=None):
    """Stand-in engine turn: a switch lane installs its bench member, a move lane deals a fifth of the target's max HP
    (fixed, no PP use), a faint forces a replacement, an empty side loses. Trackers follow through the real public
    hooks: the outgoing/damaged active is observed (legacy _observe) and each installation runs hooks.install
    (its own swap-with-slot-0 own_order rule)."""
    from gpu_engine import engine_device_public_hooks as hooks, engine_device_trackers as trackers_module, transition
    from gpu_engine.state import bench_indices
    b = state.done.shape[0];r = j.arange(b)[:, None];sides = j.arange(2)[None, :];live = ~state.done
    hp_active = state.hp[r, sides, state.active];forced = state.forced_switch | (hp_active <= 0);any_forced = j.any(forced, axis=1)
    target = j.take_along_axis(bench_indices(state.active), j.clip(actions - 4, 0, 4)[..., None], axis=2)[..., 0]
    switching = live[:, None] & (actions >= 4) & (forced | ~any_forced[:, None])
    active = j.where(switching, target, state.active)
    attacking = live[:, None] & ~any_forced[:, None] & (actions >= 0) & (actions < 4)
    hit = attacking[:, ::-1];maximum = state.max_hp[r, sides, active];current = state.hp[r, sides, active]
    after_hp = j.where(hit, j.maximum(0, current - (maximum + 4) // 5), current)
    hp = state.hp.at[r, sides, active].set(after_hp)
    out = ~j.any(hp > 0, axis=2);done = state.done | (live & j.any(out, axis=1))
    winner = j.where(live & out[:, 0] & out[:, 1], 2, j.where(live & out[:, 0], 1, j.where(live & out[:, 1], 0, state.winner)))
    turn = state.turn + (live & ~any_forced).astype(j.int32)
    after = state._replace(active=active, hp=hp, forced_switch=live[:, None] & ~done[:, None] & hit & (after_hp <= 0), done=done,
        winner=winner, turn=turn)
    t = trackers
    for side in range(2):t = trackers_module._observe(t, after, side, state.active[:, side], live, identity=False)
    c = hooks.Context(t, belief_tables, None)
    for side in range(2):c = hooks.install(c, after, side, live & (active[:, side] != state.active[:, side]))
    t = c.trackers._replace(turn=j.where(live & ~any_forced, turn + 1, c.trackers.turn))
    return hooks.PublicStepResult(transition.StepResult(after, j.zeros(b, bool), j.zeros(b, j.int32)), t)


SCRIPT = {1: 4, 2: 5, 3: 4}   # public turn -> preferred request lane (switch lanes 4..8); otherwise move lane 0


def scripted_forward(params, inputs):
    """Logits keyed by the public turn (TURN_COUNT = turn / 1000): turns 1-3 switch 0 -> 1 -> 2 -> 0 (lanes 4, 5, 4 in
    request order) at logit 30, so sampling follows the script; later turns prefer move 0 at logit 3 only, so the
    remaining lanes keep real mass. Two POSITIONAL terms make the policy depend on bench order, as a network with
    request-slot features may: move 0 reads the own token at request position 1 and switch lane 0 the one at position 2
    (their BASE_ATK). An identity-aligned comparison of two bench orders therefore shows a nonzero effect."""
    from gpu_engine import v5_model, engine_device_observation as encoder
    x = inputs['numeric_features'][:, 0];n = x.shape[0];atk = encoder._index('BASE_ATK')
    turn = j.round(x[:, 0, encoder._index('TURN_COUNT')] * 1000).astype(j.int32)
    lane = j.select([turn == t for t in SCRIPT], [j.full(n, v, j.int32) for v in SCRIPT.values()], j.zeros(n, j.int32))
    sharp = j.where((turn >= 1) & (turn <= 3), 30., 3.)
    logits = sharp[:, None] * (j.arange(9)[None, :] == lane[:, None]) + 0. * params['w']
    logits = logits.at[:, 0].add(2. * x[:, 2, atk]).at[:, 4].add(3. * x[:, 3, atk])
    return v5_model.Output(logits.astype(j.float32), j.zeros(n, j.float32), None, j.ones(n, bool))


def smoke_setup(f, *, games, shard, shards, world_blocks=1, step_cache=None):
    """A parity.Setup over the fixture: real make_step/executor/world code, the scripted step and forward."""
    from pokezero import v5_gumbel as original
    from gpu_engine import v2_search_head_to_head as h2h, device_public_teacher as teacher, engine_device_public_hooks as hooks
    from gpu_engine import v5_device_rollout as rollout
    engine = SimpleNamespace(combat=None)
    picked = parity.shard_games(tuple(range(len(games))), shard, shards);n = len(picked)
    rows = [games[g] for g in picked] * 2   # the swapped copy keeps the teams per seat; A moves to seat 1 (a_side)
    if step_cache is None or 'step' not in step_cache:
        step = h2h.make_step(engine, f.tables, None, sampling_seed=11, max_requests=64, forwards=(scripted_forward, scripted_forward))
        if step_cache is not None:step_cache['step'] = step
    else:step = step_cache['step']
    def fresh():
        state, truth = teams_state(f, rows)
        initialized = hooks.initialize(state, f.tables.belief)
        streams = np.asarray([100 + g for g in picked] + [500 + g for g in picked], np.int32)
        return rollout.initialize((state, initialized.trackers), streams, truth=truth)
    executor = (step_cache or {}).get('executor') or teacher.make_executor(engine, f.tables, enable_device_teacher=True)
    if step_cache is not None:step_cache['executor'] = executor
    recipes = tuple(original.Recipe(s) for s in h2h.world_block_seeds(2026110500, world_blocks))
    params = {'w': j.zeros((), j.float32)}
    return parity.Setup(executor, f.compiled, f.materializer, recipes, step[0], step[1], (params, params),
        np.repeat(np.asarray([0, 1], np.int32), n), np.tile(np.asarray(picked, np.int32), 2), np.repeat(np.asarray([0, 1], np.int32), n),
        fresh, dict(shard=shard, shards=shards, config='synthetic', config_sha256='0' * 64, iteration=1, block=0, max_games=len(games),
            games=list(picked), world_blocks=world_blocks, search_seed=2026110500), forward=scripted_forward)


# ---------------------------------------------------------------------------------------------- pure pieces

def tracker_state(b=1, active=0):
    from typing import NamedTuple
    class S(NamedTuple):
        active: object;species: object;level: object;gender: object;hp: object;max_hp: object;status: object
        boosts: object;turn: object;weather: object;spikes: object
    mon = lambda v: j.full((b, 2, 6), v, j.int32)
    return S(j.full((b, 2), active, j.int32), mon(1), mon(80), mon(0), mon(100), mon(100), mon(0), j.zeros((b, 2, 6, 7), j.int32),
        j.zeros(b, j.int32), j.zeros(b, j.int32), j.zeros((b, 2), j.int32))


def swap_history(switches):
    """The real legacy tracker (engine_device_trackers.initialize/update) through a switch sequence of side 1."""
    from gpu_engine import engine_device_trackers as trackers
    state = tracker_state();t = trackers.initialize(state)
    for incoming in switches:
        after = state._replace(active=state.active.at[0, 1].set(incoming))
        t = trackers.update(t, state, after, j.zeros((1, 2), j.int32));state = after
    return [int(x) for x in np.asarray(t.own_order)[0, 1]]


def rebase_rule(active, assignment):
    """The offline foe order from the offline code itself: device_public_world_state._rebase_host on a one-row carry."""
    from gpu_engine import device_public_world_state as states, engine_device_trackers as trackers
    t = jax.device_get(trackers.initialize(tracker_state()))
    root = SimpleNamespace(public=t, address=SimpleNamespace(owner=0))
    hp = np.full((1, 2, 6), 100, np.int32)
    out = states._rebase_host(root, assignment, active, state=SimpleNamespace(hp=hp, max_hp=hp))
    return [int(x) for x in np.asarray(out.own_order)[0, 1]]


def obs(seed=0):
    rng = np.random.default_rng(seed)
    return dict(numeric_features=rng.random((29, 132)).astype(np.float32), categorical_ids=rng.integers(0, 50, (29, 41)).astype(np.int32),
        token_type_ids=np.zeros(29, np.int32), attention_mask=np.ones(29, bool), legal_action_mask=rng.random(9) < .5)


def encode_order(base, order, active):
    """A synthetic view whose own tokens 1-6 and switch tokens/lanes follow `order` (the encoder's gathers), member p's
    token content being base's token 1 + p; switch token content base's 17 + member index (mod 5)."""
    out = {k: np.array(v, copy=True) for k, v in base.items()}
    for k, p in enumerate(order):
        for f in ('numeric_features', 'categorical_ids'):out[f][1 + k] = base[f][1 + p]
    for i, p in enumerate(parity.switch_members(order, active)):
        for f in ('numeric_features', 'categorical_ids'):out[f][17 + i] = base[f][23 + p % 6]
        out['legal_action_mask'][4 + i] = p % 2 == 0
    lanes = list(parity.LANE_CONSTANT_COLUMNS)   # secondary/role/slot categories stay with the lane
    out['categorical_ids'][13:22, lanes] = base['categorical_ids'][13:22, lanes]
    return out


class OrderRuleTests(unittest.TestCase):
    def test_tracker_swaps_with_slot_zero_and_the_offline_rule_sorts(self):
        # The exact back-switch example: 0 -> 1 -> 2 -> 0.
        self.assertEqual(swap_history([1]), [1, 0, 2, 3, 4, 5])
        self.assertEqual(swap_history([1, 2]), [2, 0, 1, 3, 4, 5])
        self.assertEqual(swap_history([1, 2, 0]), [0, 2, 1, 3, 4, 5])
        self.assertEqual(rebase_rule(0, {0: 0, 1: 1, 2: 2}), [0, 1, 2, 3, 4, 5])
        self.assertEqual(parity.rule_order(0), rebase_rule(0, {0: 0, 1: 1, 2: 2}))
        self.assertNotEqual(swap_history([1, 2, 0]), parity.rule_order(0))
        # They agree on in-order first switch-ins (1, then 2) ...
        self.assertEqual(swap_history([1]), parity.rule_order(1));self.assertEqual(swap_history([1, 2]), parity.rule_order(2))
        # ... but not on a first switch-in that skips a party slot (static finding): 0 -> 3 is [3, 1, 2, 0, 4, 5].
        self.assertEqual(swap_history([3]), [3, 1, 2, 0, 4, 5]);self.assertEqual(rebase_rule(3, {0: 0, 1: 3}), [3, 0, 1, 2, 4, 5])

    @SHOWDOWN
    def test_live_hooks_install_uses_the_same_swap(self):
        from gpu_engine import engine_device_public_hooks as hooks
        f = fixture();state, _ = teams_state(f, [[team(f, SPECIES[:6], [1] * 6), team(f, SPECIES[1:], [1] * 6)]])
        c = hooks.Context(hooks.initialize(state, f.tables.belief).trackers, f.tables.belief, None)
        for incoming in (1, 2, 0):
            state = state._replace(active=state.active.at[0, 1].set(incoming))
            c = hooks.install(c, state, 1, j.ones(1, bool))
        self.assertEqual([int(x) for x in np.asarray(c.trackers.own_order)[0, 1]], [0, 2, 1, 3, 4, 5])

    def test_layout_and_revealed_positions(self):
        mapping = [0, 2, -1, 1, -1, -1]   # physical 0 at slot 0, physical 3 at slot 1, physical 1 at slot 2
        self.assertEqual(parity.offline_layout_order(3, mapping), [3, 0, 1, 2, 4, 5])
        self.assertEqual(parity.offline_layout_order(0, mapping), [0, 3, 1, 2, 4, 5])
        self.assertFalse(parity.revealed_positions_differ([3, 0, 1, 2, 4, 5], [3, 0, 1, 2, 4, 5], mapping))
        self.assertTrue(parity.revealed_positions_differ([3, 1, 2, 0, 4, 5], parity.offline_layout_order(3, mapping), mapping))
        # Hidden members' positions are not compared.
        self.assertFalse(parity.revealed_positions_differ([3, 0, 1, 2, 5, 4], parity.offline_layout_order(3, mapping), mapping))
        self.assertEqual(parity.switch_members([2, 0, 1, 3, 4, 5], 2), [0, 1, 3, 4, 5])
        self.assertEqual(parity.switch_members([0, 2, 1, 3, 4, 5], 2), [0, 1, 3, 4, 5])


class DiffAlignTests(unittest.TestCase):
    def test_diff_lists_every_cell_with_names(self):
        a = obs();b = {k: np.array(v, copy=True) for k, v in a.items()}
        b['numeric_features'][0, 115] += 1   # field SELF_HAZARD_CREDIT
        b['categorical_ids'][8, 4] += 1       # opponent TYPE_1
        b['legal_action_mask'][6] = ~b['legal_action_mask'][6]
        b['attention_mask'][3] = False
        d = parity.diff(a, b)
        self.assertEqual(d['total'], 4)
        self.assertEqual(d['numeric']['cells'], [[0, 115, round(float(a['numeric_features'][0, 115]), 6), round(float(b['numeric_features'][0, 115]), 6)]])
        self.assertEqual(sorted(d['keys']), sorted(['field:numeric:SELF_HAZARD_CREDIT', 'opponent:categorical:TYPE_1', 'legal:legal:lane_6', 'own:attention:']))
        self.assertEqual(parity._class_of('field:numeric:SELF_HAZARD_CREDIT'), 'peer_belief_columns')
        self.assertEqual(parity._class_of('own:categorical:TYPE_1'), 'types')
        self.assertEqual(parity._class_of('legal:legal:lane_6'), 'legal')   # no cause known: its own bucket
        self.assertEqual(parity._class_of('legal:legal:lane_1/pp'), 'peer_belief_columns')
        self.assertEqual(parity._class_of('legal:legal:lane_5/trap'), 'carry_fields')
        self.assertEqual(parity._class_of('legal:legal:lane_5/foe_ability_trap'), 'legal')
        self.assertEqual(parity.with_causes(d, {6: 'pp'})['keys'][1], 'legal:legal:lane_6/pp')
        self.assertEqual(parity._class_of('move_actions:numeric:MOVE_PP_FRACTION'), 'peer_belief_columns')
        self.assertEqual(parity._class_of('own:numeric:ACTUAL_ATK'), 'other')
        self.assertNotIn('token_type', parity.serial(d));self.assertEqual(parity.serial(d)['numeric']['count'], 1)
        self.assertEqual(parity.diff(a, a)['total'], 0)
        self.assertEqual(parity.column_name(23, 'numeric', 130), 'reason');self.assertEqual(parity.column_name(12, 'categorical', 40), 'TRACED_ABILITY')
        self.assertEqual(parity.column_name(1, 'categorical', 10), 'BELIEF_ABILITY_1');self.assertEqual(parity.column_name(1, 'categorical', 33), 'VOLATILE_0')

    def test_alignment_removes_exactly_a_bench_permutation(self):
        base = obs(1);real_order, rule = [0, 2, 1, 3, 4, 5], parity.rule_order(0);lanes = list(parity.LANE_CONSTANT_COLUMNS)
        real, world = encode_order(base, real_order, 0), encode_order(base, rule, 0)
        raw = parity.diff(world, real);self.assertGreater(raw['total'], 0)
        self.assertEqual({k.split(':')[0] for k in raw['keys']}, {'own', 'switch_actions', 'legal'})
        aligned = parity.diff(parity.align(world, rule, real_order, 0), real)
        self.assertEqual(aligned['total'], 0)
        # An extra non-order difference survives the alignment.
        world['numeric_features'][0, 5] += 1
        aligned = parity.diff(parity.align(world, rule, real_order, 0), real)
        self.assertEqual(aligned['keys'], ['field:numeric:TURN_COUNT'])
        # Move lanes by a {world slot: real slot} map.
        moved = {k: np.array(v, copy=True) for k, v in real.items()}
        for name in ('numeric_features', 'categorical_ids'):moved[name][[13, 14]] = real[name][[14, 13]]
        moved['categorical_ids'][13:15, lanes] = real['categorical_ids'][13:15, lanes]
        moved['legal_action_mask'][[0, 1]] = real['legal_action_mask'][[1, 0]]
        self.assertEqual(parity.diff(parity.align(moved, real_order, real_order, 0, move_map={0: 1, 1: 0, 2: 2, 3: 3}), real)['total'], 0)

    def test_pre_code_follows_the_frame_pricer(self):
        own = dict(obs(2), error=np.int32(0));inputs = {k: own[k] for k in parity.OBS_FIELDS[:4]}
        legal = own['legal_action_mask']
        self.assertEqual(parity.pre_code(own, dict(error=0), inputs, legal), 0)
        self.assertEqual(parity.pre_code(dict(own, error=np.int32(4)), None, inputs, legal), 1)
        self.assertEqual(parity.pre_code(own, None, inputs, ~legal), 2)
        changed = dict(inputs, categorical_ids=inputs['categorical_ids'] + 1)
        self.assertEqual(parity.pre_code(own, None, changed, legal), 4)
        self.assertEqual(parity.pre_code(own, dict(error=1), inputs, legal), 8)


class RedactionTests(unittest.TestCase):
    def test_hp_from_public_percent_is_the_materializer_formula(self):
        # Python round (half to even), the floor of one HP, a shown-0 faint with its status cleared, hidden members kept.
        hp, status = parity.foe_hp_from_public_percent([300, 2, 1, 0, 7, 99], [301, 3, 50, 80, 5, 99], [4, 0, 0, 1, 0, 0],
            [0, 1, 2, 3, 4, -1], [37, 50, 1, 0, 50, 0])
        self.assertEqual(hp.tolist(), [111, 2, 1, 0, 2, 99])   # round(111.37), round(1.5)=2, max(1, round(.5)=0), 0, round(2.5)=2
        self.assertEqual(status.tolist(), [4, 0, 0, 0, 0, 0])

    @SHOWDOWN
    def test_tiers_take_only_the_named_fields_of_the_offline_redaction(self):
        from gpu_engine import device_public_worlds as worlds, engine_device_public_hooks as hooks
        f = fixture();state, _ = teams_state(f, [[team(f, SPECIES[:6], [1] * 6), team(f, SPECIES[1:], [1] * 6)]])
        t = hooks.initialize(state, f.tables.belief).trackers
        t = t._replace(damage_exact_taken=t.damage_exact_taken + 1., damage_exact_dealt=t.damage_exact_dealt + 2.,
            public_port=t.public_port._replace(hazards=t.public_port.hazards + .25),
            damage_history=t.damage_history._replace(hp=t.damage_history.hp + 3))
        owner = 0;foe = 1
        redacted = jax.device_get(worlds._redact_peer_private(t, owner));clone = jax.device_get(t)
        tiers = parity.redaction_tiers(clone, redacted)
        np.testing.assert_array_equal(tiers['raw'].damage_exact_taken, clone.damage_exact_taken)
        np.testing.assert_array_equal(tiers['peer_damage_exact'].damage_exact_taken[:, foe], 0)
        np.testing.assert_array_equal(tiers['peer_damage_exact'].damage_exact_dealt[:, owner], 0)
        np.testing.assert_array_equal(tiers['peer_damage_exact'].damage_exact_taken[:, owner], 1)
        np.testing.assert_array_equal(tiers['peer_damage_exact'].public_port.hazards, clone.public_port.hazards)
        np.testing.assert_array_equal(tiers['peer_exact_hazard'].public_port.hazards[:, foe, 1], 0)
        np.testing.assert_array_equal(tiers['peer_exact_hazard'].public_port.hazards[:, foe, 0], .25)
        np.testing.assert_array_equal(tiers['peer_exact_hazard'].public_port.hazards[:, owner], .25)
        # The last tier equals the offline redaction except the identity bookkeeping _rebase re-sets.
        last = tiers['peer_exact_hazard']
        for name in type(last)._fields:
            if name in ('physical_to_public', 'active_physical', 'own_order'):continue
            for x, y in zip(jax.tree.leaves(getattr(last, name)), jax.tree.leaves(getattr(redacted, name))):np.testing.assert_array_equal(x, y, err_msg=name)


class SelectionShardTests(unittest.TestCase):
    def test_selection_is_seeded_and_shard_independent(self):
        picks = [parity.selected(7, s, fr, o, .25) for s in range(40) for fr in range(10) for o in (0, 1)]
        self.assertEqual(picks, [parity.selected(7, s, fr, o, .25) for s in range(40) for fr in range(10) for o in (0, 1)])
        self.assertTrue(.18 < np.mean(picks) < .32)
        self.assertNotEqual(picks, [parity.selected(8, s, fr, o, .25) for s in range(40) for fr in range(10) for o in (0, 1)])
        self.assertTrue(parity.selected(7, 1, 2, 0, 1.))
        games = tuple(range(10))
        shards = [parity.shard_games(games, i, 3) for i in range(3)]
        self.assertEqual(sorted(g for s in shards for g in s), list(games));self.assertEqual(shards[1], (1, 4, 7))
        self.assertEqual(parity.shard_quota(2000, 3), 667)
        with self.assertRaises(ValueError):parity.shard_games(games, 3, 3)


# ---------------------------------------------------------------------------------------------- the true world

@SHOWDOWN
class TrueWorldTests(unittest.TestCase):
    """O = 0 sees foe members physical 0 (slot 0) and 3 (slot 1, active): materializer order [0, 3, 1, 2, 4, 5]."""
    def context(self, foe_team, *, switches=(3,)):
        from gpu_engine import engine_device_public_hooks as hooks, device_public_worlds as worlds
        from gpu_engine.v5_device_rollout import RootContext
        f = fixture();state, truth = teams_state(f, [[team(f, SPECIES[:6], [1] * 6), foe_team]])
        c = hooks.Context(hooks.initialize(state, f.tables.belief).trackers, f.tables.belief, None)
        for incoming in switches:
            state = state._replace(active=state.active.at[0, 1].set(incoming));c = hooks.install(c, state, 1, j.ones(1, bool))
        context = RootContext(state, c.trackers, j.asarray([81], j.int32), j.asarray([4], j.int32))
        root = worlds.detach_roots(context, f.compiled, rows=(0,), owners=(0,), enable_device_worlds=True)[0]
        host = jax.device_get(state);clone = jax.device_get(c.trackers)
        real = dict(species=host.species[0, 1], level=host.level[0, 1], moves=host.moves[0, 1], base_ability=host.base_ability[0, 1],
            item=host.item[0, 1], gender=host.gender[0, 1], hidden_power_type=host.hidden_power_type[0, 1],
            physical_to_public=clone.physical_to_public[0, 1], active=int(host.active[0, 1]),
            truth_probabilities=np.asarray(truth.probabilities)[0, 1], truth_known=np.asarray(truth.known)[0, 1])
        return f, root, real, host, clone

    def build(self, f, root, real):
        from pokezero import v5_gumbel as original
        from gpu_engine import device_public_world_state as states
        recipe = original.Recipe(17);keys = dict(zip([states.kernel_seed(root, 0, recipe)], states.world_keys([states.kernel_seed(root, 0, recipe)])))
        return parity.build_true_world(root, real, f.compiled, f.materializer, recipe, keys)

    def test_real_coordinates_mapping_rule_and_move_slots(self):
        f = fixture()
        foe = team(f, SPECIES[1:], [1, 0, 1, 0, 1, 1], permutation=(3, 2, 1, 0))   # real slots reverse the variant order
        f, root, real, host, clone = self.context(foe)
        world, state, public, sigma, move_maps, notes = self.build(f, root, real)
        self.assertEqual(notes, [])
        self.assertEqual(sigma, (0, 3, 1, 2, 4, 5))
        self.assertEqual(move_maps[0], {0: 3, 1: 2, 2: 1, 3: 0})
        for name in ('species', 'level', 'moves', 'pp', 'max_hp', 'stats', 'item', 'base_ability', 'types'):
            np.testing.assert_array_equal(getattr(state, name)[0, 1], getattr(host, name)[0, 1], err_msg=name)
        np.testing.assert_array_equal(np.asarray(public.physical_to_public)[0, 1], clone.physical_to_public[0, 1])
        self.assertEqual(int(state.active[0, 1]), 3)
        self.assertEqual([int(x) for x in np.asarray(public.own_order)[0, 1]], [3, 0, 1, 2, 4, 5])   # the offline rule
        self.assertEqual([int(x) for x in clone.own_order[0, 1]], [3, 1, 2, 0, 4, 5])                 # the real swap
        # The offline layout world itself: revealed members first, then the hidden ones.
        np.testing.assert_array_equal(np.asarray(world.state.species)[0, 1], host.species[0, 1][list(sigma)])
        self.assertEqual([sigma[int(x)] for x in np.asarray(world.public.own_order)[0, 1]], [3, 0, 1, 2, 4, 5])
        # Owner cells are the real ones.
        for name in ('species', 'hp', 'moves', 'pp', 'active'):
            np.testing.assert_array_equal(getattr(state, name)[0, 0], getattr(host, name)[0, 0], err_msg=name)

    def test_a_real_set_outside_the_source_or_an_unknown_gender_is_skipped(self):
        f = fixture();foe = team(f, SPECIES[1:], [1] * 6)
        foe[2] = dict(foe[2], ability=1)   # Insomnia with variant 1's Leftovers set: no such source row
        f, root, real, host, clone = self.context(foe)
        with self.assertRaisesRegex(parity.Skip, 'no matching source row'):self.build(f, root, real)
        real['truth_known'] = np.zeros(6, bool)   # without truth: every source row of the species, still none
        with self.assertRaisesRegex(parity.Skip, 'no matching source row'):self.build(f, root, real)
        f, root, real, host, clone = self.context(team(f, SPECIES[1:], [1] * 6))
        self.assertEqual(self.build(f, root, real)[5], [])
        real['gender'] = np.full(6, -1, np.int32)
        with self.assertRaisesRegex(parity.Skip, 'gender unknown'):self.build(f, root, real)


# ---------------------------------------------------------------------------------------------- CPU end-to-end smoke

@SHOWDOWN
class EndToEndSmokeTest(unittest.TestCase):
    MAX_STEPS = 9

    def test_two_shards_and_report_on_a_consistent_game(self):
        from gpu_engine import transition
        f = fixture()
        games = [[team(f, SPECIES[:6], [1, 0, 1, 1, 0, 1]), team(f, SPECIES[1:], [0, 1, 1, 0, 1, 1], permutation=(1, 0, 3, 2))],
                 [team(f, SPECIES[1:], [1] * 6, permutation=(2, 3, 0, 1)), team(f, SPECIES[:6], [0] * 6)]]
        cache = {}
        with tempfile.TemporaryDirectory() as tmp, patch.object(transition, 'step_with_public', scripted_step):
            outs = []
            for shard in (0, 1):
                setup = smoke_setup(f, games=games, shard=shard, shards=2, world_blocks=1, step_cache=cache)
                out = Path(tmp) / ('shard-%d' % shard);outs.append(out)
                meta = parity.run_shard(setup, out, max_steps=self.MAX_STEPS, quota=parity.shard_quota(1000, 2), sample_rate=1., sample_seed=3)
                self.assertEqual(meta['steps'], self.MAX_STEPS)
                for name in ('summary.json', 'summary.md', 'positions.jsonl'):self.assertTrue((out / name).exists(), name)
            merged = parity.report([str(o) for o in outs], Path(tmp) / 'merged')
            shard_counts = [json.loads((o / 'summary.json').read_text())['decisions'] for o in outs]
            positions = [json.loads(x) for x in (Path(tmp) / 'merged' / 'positions.jsonl').read_text().splitlines()]
            with self.assertRaises(SystemExit):parity.report([str(outs[0]), str(outs[0])], Path(tmp) / 'twice')
        self.assertEqual(merged['decisions'], sum(shard_counts));self.assertEqual(len(positions), merged['decisions'])
        self.assertGreater(merged['decisions'], 10)
        # Self-checks: the pricer's views re-encode decide's observations; the Request's root inputs are the actor's own.
        self.assertEqual(merged['self_check']['owner_reencode_mismatch'], 0);self.assertEqual(merged['self_check']['foe_reencode_mismatch'], 0)
        self.assertEqual(merged['self_check']['true_world_notes'], {})
        self.assertEqual(merged['root_source']['structural'], dict(inputs_mismatch=0, legal_mismatch=0))
        forward = merged['root_source']['forward']   # the scripted forward on the Request's inputs re-derives decide's logits
        self.assertEqual((forward['checked'], forward['equal'], forward['above_tolerance']), (merged['decisions'], merged['decisions'], 0))
        self.assertEqual(merged['self_check']['materializer_public_differs'], 0)
        self.assertEqual({k: v['worlds'] for k, v in merged['owner']['classes'].items()}, {k: 0 for k in parity.OWNER_CLASSES})
        self.assertEqual(merged['owner']['would_be_pre_refused'], 0)
        # Owner worlds of a consistent game are exact and admitted (pre code 0).
        self.assertGreater(merged['owner']['compared'], 0);self.assertEqual(merged['owner']['exact'], merged['owner']['compared'])
        self.assertEqual(set(merged['owner']['pre_codes']), {'0'})
        self.assertEqual(merged['owner']['true_world']['exact'], merged['owner']['true_world']['compared'])
        # Peer: after the named redactions and the bench alignment nothing differs; HP redaction does work after hits.
        peer = merged['peer'];self.assertGreater(peer['compared'], 0)
        self.assertEqual(peer['aligned_remaining_differs'], 0, peer['remaining_cells_ranked'])
        self.assertGreater(peer['redactions']['foe_hp_public_percent']['cells'], 0)
        self.assertEqual(sum(r['introduced_cells'] for r in peer['redactions'].values()), 0)
        # The scripted 0 -> 1 -> 2 -> 0 makes the foe order [0, 2, 1, ...]: a back-switch, order-only in the peer view.
        flagged = [p for p in positions if p['order']['differs_real_coords'] and p['step'] <= 3]   # the scripted steps
        self.assertTrue(flagged)
        self.assertTrue(all(p['order']['real'][:3] == [0, 2, 1] for p in flagged))
        for p in positions:
            self.assertEqual(p['order']['differs_real_coords'], p['order']['real'] != parity.rule_order(p['order']['real'][0]))
            if p['peer']['status'] == 'compared':
                self.assertEqual(p['peer']['order_only'], bool(p['peer']['order_keys']))
        self.assertGreater(peer['order_only'], 0);self.assertEqual(peer['order_and_other'], 0)
        self.assertEqual(merged['classes']['peer_order']['verdict'], 'PRESENT')
        for name in ('types', 'root_source'):self.assertEqual(merged['classes'][name]['verdict'], 'ABSENT', name)
        # No Baton Pass, Trace, lock, trap, Pressure PP, hazard, toxic or sleep-clause state in this game: not ABSENT.
        for name in ('peer_belief_columns', 'carry_fields'):self.assertEqual(merged['classes'][name]['verdict'], 'NOT_EXERCISED', name)
        self.assertTrue(all(r['verdict'] == 'NOT_EXERCISED' for r in merged['classes']['carry_fields']['evidence']['mechanisms'].values()))
        # History: no member re-entered before the scripted 2 -> 0, which is a back-switch.
        self.assertTrue(all(p['history']['foe']['back_switched'] == (p['order']['real'][:3] == [0, 2, 1]) for p in positions if p['step'] <= 3))
        self.assertGreater(merged['order']['foe_back_switched'], 0)
        self.assertGreater(peer['move_order_permuted'], 0)
        self.assertIn('Five-class verdict', parity.render(merged))
        # Reply-prior effect: identical inputs give KL 0 (control row, repeated forward); a bench permutation does not.
        effect = merged['peer_order_effect'];groups = {g['group']: g for g in effect['groups']}
        self.assertEqual(effect['decisions'], peer['compared'])
        self.assertLessEqual(groups['control: no order and no residual cells']['primary']['kl_max'], 1e-12)
        self.assertGreater(groups['control: no order and no residual cells']['primary']['n'], 0)
        self.assertLessEqual(effect['noise_floor']['repeat_forward']['kl_max'], 1e-12)
        self.assertEqual(effect['noise_floor']['repeat_forward']['n'], peer['compared'])
        self.assertGreater(groups['order-caused cells present (with or without residual)']['primary']['kl_max'], 1e-3)
        headline = effect['headline']['primary'];self.assertEqual(effect['headline']['group'], parity.EFFECT_GROUPS[0][0])
        self.assertEqual(headline['n'], peer['order_only']);self.assertGreater(headline['kl_max'], 1e-3)
        self.assertEqual((headline['nonfinite'], headline['excluded_support_differs']), (0, 0))
        self.assertEqual(effect['support_differs']['primary']['n'], 0)
        self.assertEqual(groups['all compared']['primary']['support_differs'], 0)
        self.assertEqual(merged['residual']['decisions'], 0)
        for p in positions:
            if p['peer']['status'] == 'compared':
                e = p['peer']['effect']['primary'];self.assertAlmostEqual(sum(e['pa']), 1., places=5)
                self.assertEqual(len(p['peer']['effect']['identities']), 9)
                if not p['peer']['order_keys'] and not p['peer']['aligned']['total']:self.assertEqual(e['pa'], e['pb'])
        self.assertIn('Reply-prior effect of the peer order', parity.render(merged))


def planted(materialize):
    """materialize_host with three planted world defects, one per detectable class: the foe's shown types zeroed in the
    world public (types), the foe's first move slot one PP short (peer belief columns) and the foe side trapped (carry)."""
    def wrapper(root, start, tables, recipe, *, keys):
        world = materialize(root, start, tables, recipe, keys=keys)
        if world.state is None:return world
        foe = 1 - root.address.owner;st = world.state
        pp = np.array(st.pp, copy=True);pp[0, foe, :, 0] = np.maximum(0, pp[0, foe, :, 0] - 1)
        trapped = np.array(st.trapped, copy=True);trapped[0, foe] = True
        shown = np.array(world.public.shown_types, copy=True);shown[0, foe] = 0
        return replace(world, state=st._replace(pp=pp, trapped=trapped), public=world.public._replace(shown_types=shown))
    return wrapper


@SHOWDOWN
class PlantedDefectSmokeTest(unittest.TestCase):
    """The same smoke with defects planted in the world builder: each detectable class turns PRESENT."""
    def test_planted_world_defects_are_reported_in_their_classes(self):
        from gpu_engine import transition, device_public_world_state as states
        f = fixture()
        games = [[team(f, SPECIES[:6], [1] * 6), team(f, SPECIES[1:], [0] * 6, permutation=(1, 0, 3, 2))]]
        with tempfile.TemporaryDirectory() as tmp, patch.object(transition, 'step_with_public', scripted_step), \
                patch.object(states, 'materialize_host', planted(states.materialize_host)):
            setup = smoke_setup(f, games=games, shard=0, shards=1, world_blocks=1)
            parity.run_shard(setup, Path(tmp) / 'out', max_steps=6, quota=100)
            summary = json.loads((Path(tmp) / 'out' / 'summary.json').read_text())
        verdicts = {k: v['verdict'] for k, v in summary['classes'].items()}
        self.assertEqual(verdicts, dict(types='PRESENT', root_source='ABSENT', peer_order='PRESENT', peer_belief_columns='PRESENT',
            carry_fields='PRESENT'))
        self.assertGreater(summary['classes']['types']['evidence']['zeroed_type_cells_owner_worlds'], 0)
        self.assertNotIn('0', summary['owner']['pre_codes'])   # every recipe world would be pre-refused by the pricer
        cells = {x['cell'] for x in summary['classes']['peer_belief_columns']['evidence']['cells']}
        self.assertIn('move_actions:numeric:MOVE_PP_FRACTION', cells)
        self.assertTrue(any(x['cell'].startswith('legal:') and x['cell'].endswith('/trap') for x in summary['classes']['carry_fields']['evidence']['cells']))
        # The zeroed shown types are a materializer public-data defect: the world public differs from the re-based one.
        self.assertGreater(summary['self_check']['materializer_public_differs'], 0)
        self.assertEqual(set(summary['self_check']['materializer_public_fields']), {'shown_types'})
        self.assertGreater(summary['owner']['classes']['types']['worlds'], 0)
        trapped = summary['classes']['carry_fields']['evidence']['mechanisms']['peer.trapped']
        self.assertEqual(trapped['verdict'], 'NOT_EXERCISED')   # the planted trap is in the world, not in the real game


def position(*, peer='compared', peer_keys=(), owner_worlds=(), peer_mechanisms=(), owner_mechanisms=()):
    """A minimal positions.jsonl record for the summary's mechanism logic."""
    aligned = dict(total=len(peer_keys), keys=sorted(peer_keys))
    kind = dict(compared='compared', refused='refused', not_requested='not_requested')[peer]
    return dict(owner_check=dict(worlds=[dict(w, refusal=w.get('refusal'), total=len(w.get('keys', ())), keys=list(w.get('keys', ())),
            aligned_keys=list(w.get('keys', ())), classes=sorted({parity._class_of(k) for k in w.get('keys', ())}), pre=0) for w in owner_worlds]),
        peer=dict(status='compared' if peer == 'compared' else 'skipped', kind=kind, reason=None if peer == 'compared' else kind, aligned=aligned),
        mechanisms=dict(peer={m: True for m in peer_mechanisms}, owner={m: True for m in owner_mechanisms}))


class MechanismVerdictTests(unittest.TestCase):
    def test_per_mechanism_verdicts_count_only_compared_decisions(self):
        P = [position(peer='refused', peer_mechanisms=('baton_pass',)),                         # exercised but refused
             position(peer='not_requested', peer_mechanisms=('baton_pass',)),                   # exercised, foe not requested
             position(peer_mechanisms=('trace',)),                                              # compared clean
             position(peer_mechanisms=('trapped',), peer_keys=('legal:legal:lane_4/trap',)),    # compared with a carry difference
             position(owner_worlds=[dict(world='b0w0', refusal='suspended/replacement')], owner_mechanisms=('baton_pass',))]
        table = parity.mechanism_table(P, 'peer', parity.CARRY_MECHANISMS, 'carry_fields')
        self.assertEqual(table['baton_pass'], dict(sampled=2, compared=0, refused=1, not_requested=1, with_class_differences=0,
            with_any_differences=0, verdict='NOT_EXERCISED'))
        self.assertEqual(table['trace']['verdict'], 'ABSENT');self.assertEqual(table['trapped']['verdict'], 'PRESENT')
        self.assertEqual(table['choice_lock']['verdict'], 'NOT_EXERCISED')
        owner = parity.mechanism_table(P, 'owner', parity.CARRY_MECHANISMS, 'carry_fields')
        self.assertEqual((owner['baton_pass']['sampled'], owner['baton_pass']['compared'], owner['baton_pass']['refused']), (1, 0, 1))
        self.assertEqual(owner['baton_pass']['verdict'], 'NOT_EXERCISED')
        # Class verdicts: no difference with some mechanisms never compared is not ABSENT.
        clean = {k: v for k, v in table.items() if k != 'trapped'}
        self.assertEqual(parity.class_verdict(0, clean, 3), 'ABSENT_WHERE_EXERCISED')
        self.assertEqual(parity.class_verdict(0, {'trace': dict(verdict='ABSENT')}, 3), 'ABSENT')
        self.assertEqual(parity.class_verdict(0, {'baton_pass': dict(verdict='NOT_EXERCISED')}, 3), 'NOT_EXERCISED')
        self.assertEqual(parity.class_verdict(0, {'trace': dict(verdict='ABSENT')}, 0), 'NOT_EXERCISED')
        self.assertEqual(parity.class_verdict(1, clean, 3), 'PRESENT')

    def test_history_back_switch_and_baton_pass(self):
        h = parity.History(np.zeros((1, 2), np.int32))
        no = np.zeros((1, 2), bool)
        h.update(np.asarray([[0, 3]]), no);self.assertEqual(h.flags(0, 1)['back_switched'], False)   # first entry out of party order
        h.update(np.asarray([[0, 3]]), np.asarray([[False, True]]))                                   # side 1 passes: replacement pending
        h.update(np.asarray([[0, 2]]), no);f = h.flags(0, 1)
        self.assertEqual((f['back_switched'], f['baton_in'], f['switch_ins'], f['ever_active']), (False, True, 2, [0, 2, 3]))
        h.update(np.asarray([[0, 3]]), no);f = h.flags(0, 1)
        self.assertEqual((f['back_switched'], f['baton_in']), (True, False))
        self.assertEqual(h.flags(0, 0)['switch_ins'], 0)

    def test_legal_causes(self):
        def state(**cells):
            base = dict(pp=np.full((1, 2, 6, 4), 5), disabled=np.zeros((1, 2, 6, 4), bool), moves=np.ones((1, 2, 6, 4), np.int32),
                choice_move=np.zeros((1, 2, 6), np.int32), encore_move=np.zeros((1, 2, 6), np.int32), charge_move=np.zeros((1, 2, 6), np.int32),
                hp=np.full((1, 2, 6), 50), trap_source=np.full((1, 2, 6), -1), volatile=np.zeros((1, 2, 6), np.uint32),
                trapped=np.zeros((1, 2), bool), ability=np.ones((1, 2, 6), np.int32), active=np.zeros((1, 2), np.int32))
            for k, v in cells.items():base[k] = v
            return SimpleNamespace(**base)
        real = state();order = [0, 2, 1, 3, 4, 5]
        pp = real.pp.copy();pp[0, 1, 0, 2] = 0
        hp = real.hp.copy();hp[0, 1, 2] = 0
        trapped = np.asarray([[False, True]])
        ability = real.ability.copy();ability[0, 0, 0] = 9
        self.assertEqual(parity.legal_causes([2], state(pp=pp), real, 1, order, 0), {2: 'pp'})
        self.assertEqual(parity.legal_causes([4], state(hp=hp), real, 1, order, 0), {4: 'hp'})   # lane 4 is member 2 in [0, 2, 1, ...]
        self.assertEqual(parity.legal_causes([5], state(trapped=trapped), real, 1, order, 0), {5: 'trap'})
        self.assertEqual(parity.legal_causes([6], state(ability=ability), real, 1, order, 0), {6: 'foe_ability_trap'})
        self.assertEqual(parity.legal_causes([7, 1], real, real, 1, order, 0), {7: 'unknown', 1: 'unknown'})

    def test_zeroed_types_count_the_full_arrays(self):
        a = obs(3);b = {k: np.array(v, copy=True) for k, v in a.items()}
        a['categorical_ids'][1:13, 4:6] = 0;b['categorical_ids'][1:13, 4:6] = 7
        a['numeric_features'] += 1   # more than CELL_CAP numeric cells: the type count must not depend on the listing cap
        d = parity.diff(a, b)
        self.assertEqual(d['zeroed_types'], 24);self.assertEqual(parity._zeroed_types(parity.serial(d)), 24)
        self.assertEqual(len(d['numeric']['cells']), parity.CELL_CAP)


class EffectTests(unittest.TestCase):
    def test_kl_tv_and_top1_flip_on_aligned_distributions(self):
        legal = np.asarray([True, True] + [False] * 7)
        a = np.asarray([0., np.log(3.)] + [5.] * 7);b = np.zeros(9)   # a: (.25, .75) on the legal lanes, b: (.5, .5)
        e = parity.effect(a, legal, b, legal)
        self.assertAlmostEqual(e['kl'], .5 * np.log(4 / 3), places=12)
        self.assertAlmostEqual(e['tv'], .25, places=12);self.assertAlmostEqual(e['max_abs_dp'], .25, places=12)
        self.assertTrue(e['top1_flip']);self.assertEqual((e['top_a'], e['top_b']), (1, 0))   # b ties at lanes 0/1: lowest lane
        self.assertEqual(e['legal'], 2);self.assertFalse(e['support_differs'])
        self.assertEqual(e['pa'][2:], [0.] * 7)
        same = parity.effect(a, legal, a, legal)
        self.assertEqual((same['kl'], same['tv'], same['top1_flip'], same['max_abs_dp']), (0., 0., False, 0.))
        # Different legal sets: compared on the common lanes, flagged.
        wide = legal.copy();wide[4] = True
        e = parity.effect(a, wide, b, legal);self.assertTrue(e['support_differs']);self.assertAlmostEqual(e['kl'], .5 * np.log(4 / 3), places=12)
        self.assertIsNone(parity.effect(a, legal, b, ~legal))
        self.assertTrue(np.isneginf(parity.masked_log_softmax([1., 2.], [True, False])[1]))

    def test_switch_lanes_align_across_a_permuted_bench(self):
        real_order, rule = [0, 2, 1, 3, 4, 5], parity.rule_order(0)
        perm = parity.lane_permutation(rule, real_order, 0)
        self.assertEqual(perm, [0, 1, 2, 3, 5, 4, 6, 7, 8])
        # The same mapping as align() on the legal lanes, also for a non-front active and a move-slot map.
        base = obs(4)
        for world_order, order, active, moves in ((rule, real_order, 0, None), ([3, 0, 1, 2, 4, 5], [3, 1, 2, 0, 4, 5], 3, {0: 2, 1: 0, 2: 1, 3: 3})):
            perm = parity.lane_permutation(world_order, order, active, move_map=moves)
            np.testing.assert_array_equal(parity.align(base, world_order, order, active, move_map=moves)['legal_action_mask'],
                base['legal_action_mask'][perm])
            # Each real switch lane's member is the world lane's member.
            world_members, real_members = parity.switch_members(world_order, active), parity.switch_members(order, active)
            self.assertEqual([world_members[perm[4 + i] - 4] for i in range(5)], real_members)

    def test_zero_order_difference_control_and_an_equivariant_policy(self):
        rng = np.random.default_rng(5);logits = rng.normal(size=9);legal = rng.random(9) < .7;legal[[0, 4, 5]] = True   # lanes 4/5 swap below
        e = parity.effect(logits, legal, logits, legal)
        self.assertEqual((e['kl'], e['tv'], e['top1_flip']), (0., 0., False))
        # A policy that follows members, not request positions: the world's lanes are the real ones permuted; aligned, KL 0.
        real_order, rule = [0, 2, 1, 3, 4, 5], parity.rule_order(0);perm = parity.lane_permutation(rule, real_order, 0)
        world_logits, world_legal = np.empty(9), np.empty(9, bool)
        world_logits[perm], world_legal[perm] = logits, legal
        e = parity.effect(world_logits[perm], world_legal[perm], logits, legal)
        self.assertEqual((e['kl'], e['top1_flip']), (0., False))
        # Unaligned, the same policy differs whenever the permuted lanes carry different mass.
        self.assertGreater(parity.effect(world_logits, world_legal, logits, legal)['tv'], 0)

    def test_effect_summary_groups(self):
        def p(kl, *, back, rule, order, residual=0):
            e = dict(kl=kl, tv=kl / 2, top1_flip=kl > .5, max_abs_dp=kl / 3, support_differs=False, top_a=0, top_b=1 if kl > .5 else 0,
                top_a_identity='x', top_b_identity='y')
            return dict(stream=1, frame=2, owner=0, turn=3, history=dict(foe=dict(back_switched=back)), order=dict(differs_real_coords=rule),
                peer=dict(status='compared', order_keys=['own:numeric:HP_FRACTION'] if order else [], aligned=dict(total=residual),
                    effect=dict(primary=e, layout=e, noise=dict(kl=0., tv=0., top1_flip=False, max_abs_dp=0.))))
        P = [p(0., back=False, rule=False, order=False), p(.2, back=True, rule=True, order=True), p(.9, back=True, rule=True, order=True),
             p(.1, back=False, rule=True, order=True, residual=2)]
        s = parity.summarize_effect(P);groups = {g['group']: g['primary'] for g in s['groups']}
        self.assertEqual(groups['all compared']['n'], 4);self.assertEqual(groups['foe back-switched']['n'], 2)
        self.assertAlmostEqual(groups['foe back-switched']['kl_median'], .55);self.assertEqual(groups['foe back-switched']['top1_flip_share'], .5)
        self.assertEqual(groups['control: no order and no residual cells']['n'], 1);self.assertEqual(groups['control: no order and no residual cells']['kl_max'], 0.)
        self.assertEqual(groups['residual cells after alignment (any class)']['n'], 1);self.assertEqual(len(s['top1_flips']), 1)
        self.assertEqual(s['noise_floor']['repeat_forward']['kl_max'], 0.)

    def test_residual_member_detail_and_classification(self):
        vocab = dict(species=('', 'kecleon', 'castform', 'ditto', 'porygon2'), moves=('', 'tackle', 'transform', 'conversion'),
            abilities=('', 'colorchange', 'forecast', 'limber', 'trace'), items=('', 'leftovers'), types=('', 'normal', 'water', 'fire'))
        z = lambda *shape:np.zeros((1, 2) + shape, np.int32)
        st = SimpleNamespace(species=z(6), ability=z(6), base_ability=z(6), item=z(6), moves=z(6, 4), last_move=z(6), types=z(6, 2), base_types=z(6, 2))
        for p, (sp, ab, mv) in enumerate(((1, 1, 1), (2, 2, 1), (3, 3, 2), (4, 4, 3))):
            st.species[0, 1, p] = sp;st.ability[0, 1, p] = st.base_ability[0, 1, p] = ab;st.moves[0, 1, p, 0] = mv
            st.types[0, 1, p, 0] = 2;st.base_types[0, 1, p, 0] = 1
        world = SimpleNamespace(**{k: np.array(v, copy=True) for k, v in vars(st).items()});world.types[0, 1, :, 0] = 1
        clone = SimpleNamespace(physical_to_public=np.asarray([[[0, -1, -1, -1, -1, -1], [0, 1, 2, 3, -1, -1]]]),
            shown_types=np.zeros((1, 2, 6, 2), np.int32), last_moves=np.zeros((1, 2, 6), np.int32))
        order = [0, 1, 2, 3, 4, 5]
        exact = {('categorical', 1 + p, 4) for p in range(4)} | {('numeric', 0, 5)}
        out = parity.describe_residual(exact, st, world, clone, 1, order, vocab)
        kinds = [x.get('classification') for x in out if x['section'] == 'own']
        self.assertEqual(kinds, ['color_change', 'forecast', 'transform_residue', 'conversion'])
        first = next(x for x in out if x['token'] == 1)
        self.assertEqual((first['species'], first['real_types'], first['world_types'], first['cells']), ('kecleon', ['water', ''], ['normal', ''], ['own:categorical:TYPE_1']))
        self.assertNotIn('classification', next(x for x in out if x['token'] == 0))


def effect_position(kl, *, back=False, order=True, residual=0, support_differs=False, dropped=None, flip=False):
    """A compared positions.jsonl record carrying only what summarize_effect reads (dropped=None: a 7cef1753 record)."""
    e = dict(kl=kl, tv=abs(kl) / 2 if np.isfinite(kl) else kl, top1_flip=flip, max_abs_dp=abs(kl) / 3 if np.isfinite(kl) else kl,
        support_differs=support_differs, top_a=0, top_b=1 if flip else 0, top_a_identity='switch:p1', top_b_identity='switch:p2',
        pa=[.5, .5], pb=[.5, .5])
    if dropped is not None:e.update(dropped_b=dropped[0], dropped_a=dropped[1])
    return dict(stream=1, frame=2, owner=0, turn=3, history=dict(foe=dict(back_switched=back)), order=dict(differs_real_coords=order),
        peer=dict(status='compared', order_only=order and not residual, order_keys=['own:numeric:HP_FRACTION'] if order else [],
            aligned=dict(total=residual), effect=dict(primary=e, layout=dict(e), noise=dict(kl=0., tv=0., top1_flip=False, max_abs_dp=0.))))


class EffectReportTests(unittest.TestCase):
    """report-side fixes, all from positions.jsonl as written by 7cef1753 (no new field needed except the dropped mass)."""
    def test_order_only_headline_splits_by_back_switch(self):
        P = [effect_position(.4, back=True), effect_position(.2, back=False),            # order only
             effect_position(.9, back=True, residual=2),                                   # order and a residual (e.g. Color Change)
             effect_position(0., order=False)]                                             # control
        s = parity.summarize_effect(P);groups = {g['group']: g['primary'] for g in s['groups']}
        self.assertEqual(s['headline']['group'], 'order only (headline): order-caused cells, no residual cells')
        self.assertEqual(s['headline']['primary']['n'], 2);self.assertAlmostEqual(s['headline']['primary']['kl_max'], .4)
        self.assertEqual(groups['order only, foe back-switched']['n'], 1);self.assertAlmostEqual(groups['order only, foe back-switched']['kl_max'], .4)
        self.assertEqual(groups['order only, no foe back-switch']['n'], 1)
        self.assertEqual(groups['order-caused cells present (with or without residual)']['n'], 3)
        self.assertEqual(groups['residual cells after alignment (any class)']['n'], 1)
        # A record without peer.order_only falls back to order-caused cells and no residual.
        old = effect_position(.3);del old['peer']['order_only']
        self.assertEqual(parity.summarize_effect([old])['headline']['primary']['n'], 1)

    def test_support_differs_excluded_with_dropped_mass(self):
        legal_a = np.asarray([True, True] + [False] * 7);legal_b = legal_a.copy();legal_b[4] = True
        logits = np.zeros(9);logits[4] = np.log(2.)   # b: lanes 0, 1, 4 at 1/4, 1/4, 1/2
        e = parity.effect(logits, legal_a, logits, legal_b)
        self.assertTrue(e['support_differs']);self.assertAlmostEqual(e['dropped_b'], .5, places=12);self.assertEqual(e['dropped_a'], 0.)
        self.assertEqual(e['kl'], 0.)   # equal on the common lanes
        same = parity.effect(logits, legal_b, logits, legal_b);self.assertEqual((same['dropped_b'], same['dropped_a']), (0., 0.))
        P = [effect_position(.1), effect_position(2., support_differs=True, dropped=(.5, 0.)), effect_position(3., support_differs=True)]
        s = parity.summarize_effect(P);head = s['headline']['primary']
        self.assertEqual((head['n'], head['excluded_support_differs']), (1, 2));self.assertAlmostEqual(head['kl_max'], .1)
        row = s['support_differs']['primary']
        self.assertEqual((row['n'], row['dropped_unavailable']), (2, 1));self.assertEqual((row['dropped_b_max'], row['dropped_a_max']), (.5, 0.))
        self.assertAlmostEqual(row['kl_max'], 3.)
        self.assertIn('legal sets differ', parity.render(dict(parity.summarize([], []), peer_order_effect=s)))

    def test_nonfinite_values_are_excluded_and_counted(self):
        P = [effect_position(.2), effect_position(float('nan')), effect_position(float('inf'), flip=True)]
        s = parity.summarize_effect(P)
        for g in s['groups'][:1] + [x for x in s['groups'] if x['group'] == 'all compared']:
            self.assertEqual((g['primary']['n'], g['primary']['nonfinite']), (1, 2), g['group'])
            self.assertAlmostEqual(g['primary']['kl_max'], .2);self.assertEqual(g['primary']['top1_flip_share'], 0.)
        self.assertEqual(s['top1_flips'], [])
        nan = parity.effect(np.full(9, np.nan), np.ones(9, bool), np.zeros(9), np.ones(9, bool))
        self.assertFalse(parity._finite_effect(nan))
        # A NaN in the stored probabilities alone also counts.
        bad = effect_position(.1);bad['peer']['effect']['primary']['pa'] = [float('nan'), 1.]
        self.assertEqual(parity.summarize_effect([bad])['headline']['primary']['nonfinite'], 1)
        # JSON round trip (positions.jsonl writes NaN) keeps the guard working.
        again = json.loads(json.dumps(P))
        self.assertEqual(parity.summarize_effect(again)['headline']['primary']['nonfinite'], 2)


class ReportTests(unittest.TestCase):
    def shard(self, root, name, shard, **meta):
        out = Path(root) / name;out.mkdir();(out / 'positions.jsonl').write_text('')
        parity.write_summary(out, [], [dict(dict(shard=shard, shards=2, config_sha256='c', sample_rate=.2, champion={'iteration': 9375},
            games=[], steps=0), **meta)])
        return out

    def test_empty_shards_merge_and_mismatches_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = self.shard(tmp, 'a', 0), self.shard(tmp, 'b', 1)
            merged = parity.report([a, b], Path(tmp) / 'merged')
            self.assertEqual(merged['decisions'], 0);self.assertEqual(merged['classes']['carry_fields']['verdict'], 'NOT_EXERCISED')
            for key, value in (('sample_rate', .3), ('champion', {'iteration': 1}), ('config_sha256', 'd')):
                other = self.shard(tmp, 'other-' + key, 1, **{key: value})
                with self.assertRaisesRegex(SystemExit, key):parity.report([a, other], Path(tmp) / ('m-' + key))


class CommandLineTests(unittest.TestCase):
    def test_run_requires_the_champion_flags_and_a_new_out(self):
        import contextlib, io
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):parity.main(['run', '--config', 'c.json', '--out', str(Path(tmp) / 'o')])
            flags = ['--champion-archive', 'a.npz', '--champion-archive-sha256', 'a' * 64, '--champion-checkpoint-sha256', 'b' * 64]
            with self.assertRaises(SystemExit):parity.main(['run', '--config', 'c.json', '--out', tmp] + flags)   # exists
            with self.assertRaises(SystemExit):parity.main(['run', '--config', 'c.json', '--out', str(Path(tmp) / 'o'), '--sample-rate', '0'] + flags)
            # An empty shard writes the full run metadata report checks.
            meta = dict(shard=3, shards=4, config='c.json', config_sha256='s', iteration=1, block=0, max_games=3, games=[], world_blocks=4,
                search_seed=7, sample_decisions=100, champion={'iteration': 9375})
            with patch.object(parity, 'build_setup', return_value=parity.EmptyShard(meta)):
                self.assertEqual(parity.main(['run', '--config', 'c.json', '--out', str(Path(tmp) / 'e'), '--shard', '3', '--shards', '4',
                    '--sample-rate', '.2', '--sample-decisions', '100'] + flags), 0)
            run = json.loads((Path(tmp) / 'e' / 'summary.json').read_text())['runs'][0]
            self.assertTrue(set(parity.REPORT_KEYS) <= set(run));self.assertEqual((run['sample_rate'], run['quota'], run['steps']), (.2, 25, 0))


class BuildSetupTests(unittest.TestCase):
    """build_setup's wiring on stand-ins of everything it loads: the --max-games prefix, the shard's games, the swapped
    copy's streams, A's seat per row and the make_step / source / executor / recipe bindings."""
    def test_prefix_shard_swap_and_bindings(self):
        from dataclasses import dataclass
        from gpu_engine import v2_fused, v2_search_head_to_head as h2h, device_public_worlds as worlds
        from gpu_engine import device_public_world_state as states, device_public_teacher as teacher, v5_device_backend
        @dataclass(frozen=True)
        class Game:
            stream_id: int;battle_id: str;team_indices: tuple
        @dataclass(frozen=True)
        class Batch:
            games: tuple;gender_root_ids: tuple;sampling_seed: int
        batch = Batch(tuple(Game(10 + g, 'b%d' % g, (g, g)) for g in range(7)), tuple(range(100, 107)), 77)
        plan = SimpleNamespace(iterations={1: [batch]}, model=SimpleNamespace(label_vocabulary=SimpleNamespace(source_binding_sha256='rows')),
            max_requests=99)
        prepared = SimpleNamespace(catalog='catalog', engine_tables='engine', observation_tables='observation', metadata_json='{"m": 1}',
            source=SimpleNamespace(metadata=SimpleNamespace(source_hash='hash')))
        constructed = []
        class Backend:
            def __init__(self, prepared_, pool, gender_policy):self.args = (prepared_, pool, gender_policy)
            def check_plan(self, plan_):constructed.append(('check', plan_))
            def _construct(self, declared):constructed.append(declared);return dict(x=j.zeros(len(declared.games)))
        made = {}
        def make_step(*args, **kwargs):made.update(args=args, kwargs=kwargs);return 'decide', 'apply'
        args = SimpleNamespace(config='c.json', iteration=1, block=0, max_games=5, shard=1, shards=2, world_blocks=2, search_seed=7)
        with patch.object(v2_fused, 'prepare', return_value=dict(plan=plan, prepared=prepared, pool='pool', gender='g', declared=dict(config_sha256='s'))), \
                patch.object(h2h, 'champion_setup', return_value=('evaluator', SimpleNamespace(evaluator={'champion': 9375}), 'params')), \
                patch.object(v5_device_backend, 'RuntimeBackend', Backend), patch.object(h2h, 'make_step', make_step), \
                patch.object(worlds, 'compile_source', return_value='source') as compile_source, \
                patch.object(states, 'compile_tables', return_value='materializer'), \
                patch.object(teacher, 'make_executor', return_value='executor'):
            setup = parity.build_setup(args)
            carry = setup.fresh()
        self.assertEqual(setup.meta['games'], [1, 3]);self.assertEqual(setup.meta['config_sha256'], 's')
        self.assertEqual(setup.a_side.tolist(), [0, 0, 1, 1]);self.assertEqual(setup.game.tolist(), [1, 3, 1, 3])
        self.assertEqual(setup.seat_order.tolist(), [0, 0, 1, 1]);self.assertEqual(setup.params, ('params', 'params'))
        first, swapped = constructed[1], constructed[2]
        self.assertEqual([g.stream_id for g in first.games], [11, 13]);self.assertEqual(first.gender_root_ids, (101, 103))
        self.assertEqual([g.stream_id for g in swapped.games], [11 - parity_swap(), 13 - parity_swap()])
        self.assertEqual([g.battle_id for g in swapped.games], ['b1-swap', 'b3-swap'])
        self.assertEqual(np.asarray(carry['x']).shape, (4,))
        self.assertEqual(made['args'], ('engine', 'observation', plan.model));self.assertEqual(made['kwargs']['forwards'], ('evaluator', 'evaluator'))
        self.assertEqual(made['kwargs']['sampling_seed'], 77);self.assertEqual(made['kwargs']['max_requests'], 99)
        self.assertEqual(compile_source.call_args.kwargs, dict(expected_source_hash='hash', expected_source_rows_sha256='rows'))
        self.assertEqual(setup.recipes, tuple(__import__('pokezero.v5_gumbel', fromlist=['Recipe']).Recipe(x) for x in h2h.world_block_seeds(7, 2)))
        # More shards than games: an empty shard.
        with patch.object(v2_fused, 'prepare', return_value=dict(plan=plan, prepared=prepared, pool='pool', gender='g', declared={})), \
                patch.object(h2h, 'champion_setup', return_value=('evaluator', SimpleNamespace(evaluator={}), 'params')), \
                patch.object(v5_device_backend, 'RuntimeBackend', Backend):
            empty = parity.build_setup(replace_ns(args, max_games=1, shard=1))
        self.assertIsInstance(empty, parity.EmptyShard);self.assertEqual(empty.meta['games'], [])
        self.assertTrue(set(parity.REPORT_KEYS) - {'sample_rate', 'sample_seed', 'policy'} <= set(empty.meta))


def parity_swap():
    from gpu_engine.v2_head_to_head import SWAP_OFFSET
    return SWAP_OFFSET


def replace_ns(ns, **values):
    return SimpleNamespace(**dict(vars(ns), **values))


if __name__ == '__main__':
    unittest.main()
