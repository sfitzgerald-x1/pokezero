"""CPU tests of scripts/depthn_opponent_view_parity.py: no GPU, no Showdown, no pricer. The live search is the real DepthNLiveSearch
(its own process_many, _batch and _search) over a fake constructor whose prepare calls phase_d_protocol_world.sample through the
module, as the real one does, with fake pricing (tests/test_depthn_rr_reprice's), so the sampler wrapper runs at the live call
site; the env, the opponent views and the true world's materialization are fakes where a real one needs Showdown or a GPU."""
from contextlib import ExitStack, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from scripts import depthn_opponent_view_parity as P
from tests import test_depthn_live_bench as BT
from tests import test_depthn_rr_reprice as RT
from gpu_engine import phase_d_protocol_world as pw, v5_public_world as vw
from gpu_engine.phase_d_live_worlds import Refusal
from pokezero import determinization as D, phase_d_owning_capture as owning
from pokezero.engine_world import unpack_team
from pokezero.showdown_fixture import pack_team

ORIGINAL_SAMPLE = pw.sample
TOKEN_TYPES = np.asarray((0,) + (1,) * 6 + (2,) * 6 + (3,) * 9 + (5,), np.int16)


# The true start -------------------------------------------------------------------------------------------------------------------
SOURCE = SimpleNamespace(
    species_metadata={name: dict(baseStats=dict(zip(('hp', 'atk', 'def', 'spa', 'spd', 'spe'), stats))) for name, stats in dict(
        snorlax=(160, 110, 65, 65, 110, 30), gengar=(60, 65, 60, 130, 75, 110), starmie=(60, 75, 85, 100, 85, 115),
        skarmory=(65, 80, 140, 40, 70, 70), zapdos=(90, 90, 85, 125, 90, 100), tyranitar=(100, 134, 110, 95, 100, 61)).items()},
    move_metadata={**{m: dict(type=t) for m, t in (('bodyslam', 'Normal'), ('return', 'Normal'), ('thunderbolt', 'Electric'), ('icepunch', 'Ice'),
        ('explosion', 'Normal'), ('surf', 'Water'), ('frustration', 'Normal'), ('psychic', 'Psychic'), ('drillpeck', 'Flying'), ('rockslide', 'Rock'),
        ('earthquake', 'Ground'))}, **{m: dict(category='Status') for m in ('curse', 'rest', 'recover', 'spikes', 'roar', 'toxic', 'agility', 'substitute',
        'dragondance')}})
FP_TEAM = (('Snorlax', 80, 'M', ('bodyslam', 'return102', 'curse', 'rest'), 'Leftovers', 'Immunity'),
    ('Gengar', 77, 'F', ('thunderbolt', 'hiddenpowerfire70', 'icepunch', 'explosion'), 'Leftovers', 'Levitate'),
    ('Starmie', 78, None, ('surf', 'frustration102', 'recover', 'psychic'), 'Leftovers', 'Natural Cure'),
    ('Skarmory', 79, 'F', ('drillpeck', 'spikes', 'roar', 'toxic'), 'Leftovers', 'Keen Eye'),
    ('Zapdos', 76, None, ('thunderbolt', 'return80', 'agility', 'substitute'), 'Leftovers', 'Pressure'),
    ('Tyranitar', 75, 'M', ('rockslide', 'earthquake', 'dragondance', 'hiddenpowerbug70'), 'Leftovers', 'Sand Stream'))
OUR_TEAM = tuple((s, l + 1, g, m, 'Leftovers', a) for s, l, g, m, _, a in reversed(FP_TEAM))


def team_rows(team):
    """self_team metadata rows (showdown._pokemon_metadata's fields) whose stats the gen3 randbat recipe reproduces."""
    rows = []
    for species, level, gender, moves, item, ability in team:
        canonical = tuple(D.canonical_move_id(m) for m in moves)
        spread = D._gen3_randbat_fixture_spread({}, species=species, moves=canonical, item=item, level=level, set_source=SOURCE)
        stats = D._computed_stats(D._base_stats_for_species(SOURCE, species), evs=spread['evs'], ivs=spread['ivs'], level=level)
        rows.append(dict(species=species, details=f'{species}, L{level}' + (f', {gender}' if gender else ''), moves=list(moves), item=item,
            ability=ability, stats={k: v for k, v in stats.items() if k != 'hp'}))
    return rows


def round0_request(rows):
    return dict(side=dict(id='p2', pokemon=[dict(details=r['details'], moves=list(r['moves'])) for r in rows]))


def public_root(rows):
    context = SimpleNamespace(observation=SimpleNamespace(metadata=dict(self_team=rows)), trajectory=SimpleNamespace(steps=()), format_id='gen3randombattle',
        player_id='p1', battle_id='battle-7', decision_round_index=3)
    return SimpleNamespace(context=context, owner_lines_sha256='e' * 64)


class TrueStartTests(unittest.TestCase):
    def test_the_true_start_is_the_samplers_format_with_foul_plays_team(self):
        ours, theirs = team_rows(OUR_TEAM), team_rows(FP_TEAM)
        start, info = P.true_start(public_root(ours), theirs, round0_request(theirs), SOURCE, 0)
        self.assertIsNone(info['reason']);self.assertEqual(set(start.player_teams), {'p1', 'p2'})
        self.assertEqual(start.observation_format_id, 'gen3randombattle')
        sampler_ours, _ = D._self_team_from_metadata_result(ours, team_size=6, set_source=SOURCE)   # what the sampler packs for us
        self.assertEqual(start.player_teams['p1'], pack_team(sampler_ours))
        mons = unpack_team(start.player_teams['p2'])
        self.assertEqual([m.species for m in mons], [t[0] for t in FP_TEAM])   # the original party order
        self.assertEqual([m.level for m in mons], [t[1] for t in FP_TEAM]);self.assertEqual([m.gender for m in mons], [t[2] for t in FP_TEAM])
        self.assertEqual([tuple(D._normalize_id(x) for x in m.moves) for m in mons], [tuple(D.canonical_move_id(x) for x in t[3]) for t in FP_TEAM])
        self.assertEqual([D._normalize_id(m.ability) for m in mons], [D._normalize_id(t[5]) for t in FP_TEAM])
        self.assertEqual([D._normalize_id(m.item) for m in mons], ['leftovers'] * 6)
        fire = dict({k: 31 for k in ('hp', 'atk', 'def', 'spa', 'spd', 'spe')}, **D._HIDDEN_POWER_IVS['fire'])
        self.assertEqual(dict(mons[1].ivs), fire)   # Gengar's Hidden Power Fire spread, as the request's stats prove
        # Happiness: Return102 proves 255 (left as the format's default), Frustration102 proves 0, Return80 is ambiguous.
        self.assertEqual(info['happiness'], [255, None, 0, None, None, None])
        self.assertEqual([e.split('|')[11] for e in start.player_teams['p2'].split(']')], ['', '', '0', '', '', ''])
        self.assertEqual(info['notes'], ['zapdos: OWN_HAPPINESS_UNAVAILABLE'])
        plain, _ = D._self_team_from_metadata_result(theirs, team_size=6, set_source=SOURCE)
        entries = pack_team(plain).split(']');fields = entries[2].split('|');fields[11] = '0';entries[2] = '|'.join(fields)
        self.assertEqual(start.player_teams['p2'], ']'.join(entries))   # pack_team's own string, field 11 of Starmie aside
        swapped, _ = P.true_start(public_root(ours), theirs, round0_request(theirs), SOURCE, 1)
        self.assertEqual((swapped.player_teams['p2'], swapped.player_teams['p1']), (start.player_teams['p1'], start.player_teams['p2']))

    def test_a_team_the_recipe_cannot_reproduce_is_refused_by_name(self):
        theirs = team_rows(FP_TEAM);theirs[3] = dict(theirs[3], stats=dict(theirs[3]['stats'], spe=theirs[3]['stats']['spe'] + 1))
        start, info = P.true_start(public_root(team_rows(OUR_TEAM)), theirs, round0_request(theirs), SOURCE, 0)
        self.assertIsNone(start);self.assertTrue(info['reason'].startswith('foul play team: '))
        start, info = P.true_start(public_root(team_rows(OUR_TEAM)[:5]), team_rows(FP_TEAM), None, SOURCE, 0)
        self.assertIsNone(start);self.assertTrue(info['reason'].startswith('own team: '))

    def test_happiness_needs_the_literal_member(self):
        rows = team_rows(FP_TEAM);request = round0_request(rows);request['side']['pokemon'][2]['details'] = 'Gengar, L78'
        packed, values, notes = P.with_happiness(pack_team(D._self_team_from_metadata_result(rows, team_size=6, set_source=SOURCE)[0]), rows, request)
        self.assertEqual(values[2], None);self.assertIn('starmie: no literal round-0 request member', notes)
        self.assertEqual(packed.split(']')[2].split('|')[11], '')

    def test_the_kernel_seed_is_the_samplers_own(self):
        root = public_root(());detached = SimpleNamespace(source_binding_sha256='b', address=SimpleNamespace(stream=5, frame=3, owner=0))
        source = SimpleNamespace(binding_sha256='b', original=None)
        with patch.object(pw, '_carried_root', lambda root, detached, lines, metadata: detached), \
                patch.object(pw, '_gen3_randbat_belief_start_override_result', return_value=(None, 'test')):
            made = pw.sample(root, detached, lines=('x',), catalog=None, source=source, metadata={}, config=None, battle_id='b', seed=20261103, count=8)
        self.assertEqual([w.kernel_seed for w in made.worlds], [P.kernel_seed(root, detached, 20261103, i) for i in range(8)])

    def test_the_true_world_is_materialized_as_the_sampler_materializes(self):
        root = public_root(());detached = SimpleNamespace(address=SimpleNamespace(stream=5, frame=3, owner=1))
        source = SimpleNamespace(original='original-source');start = object();seen = {}
        named = dict(root=root, detached=detached, lines=['a', 'b'], catalog='catalog', source=source, metadata='metadata', config='config',
            battle_id='b', seed=11, count=16)

        def materialize(*arguments, **keywords):
            seen.update(arguments=arguments, keywords=keywords)
            return pw.ProtocolWorld(None, None, ('refused',), mapper_diagnostics=(dict(kind='K', status='TRIGGERED'), dict(kind='C', status='CLEAR')))
        def true_start(*arguments):
            seen['start_arguments'] = arguments
            return start, dict(reason=None)
        with patch.object(pw, '_carried_root', lambda r, d, lines, metadata: ('carried', r, d, tuple(lines), metadata)), \
                patch.object(vw, 'materialize', side_effect=materialize), patch.object(P, 'true_start', side_effect=true_start):
            world, info = P.build_true_world(P.SampleCall(named), 'team', 'request0')
        self.assertEqual(seen['start_arguments'], (root, 'team', 'request0', 'original-source', 1))
        self.assertEqual(seen['arguments'], (root, start))
        self.assertEqual(seen['keywords'], dict(prepared=pw.ProtocolInputs(('carried', root, detached, ('a', 'b'), 'metadata'), ('a', 'b'), 'catalog', 'metadata',
            source), history=None, own_static=None, seed=P.kernel_seed(root, detached, 11, 0)))
        self.assertEqual((info['constructed'], info['refusals'], info['triggered'], info['kernel_seed']), (False, ['refused'], ['K'], P.kernel_seed(root, detached, 11, 0)))


# Overlay and views ---------------------------------------------------------------------------------------------------------------
def fp_request(snorlax='73/100', pps=(3, 4, 2), active=None, force=False):
    """Foul Play's (p2's) literal request over RT's world: its party is species 1-6 of RT.VOCABULARY, Snorlax (slot 2) active."""
    names = ('Snorlax', 'Pikachu', 'Mr. Mime', "Farfetch’d", 'Ho-Oh', 'Gengar')
    conditions = (snorlax, '100/100', '0 fnt', '0 fnt', '1/100', '2/100 par')   # all but Snorlax's equal RT.STATE's
    pokemon = [dict(ident=f'p2: {n}', details=f'{n}, L80', condition=c, active=i == 0) for i, (n, c) in enumerate(zip(names, conditions))]
    moves = active if active is not None else [dict(id=i, pp=p, maxpp=24) for i, p in zip(('thunderbolt', 'hiddenpowerfire70', 'hiddenpowerwater70'), pps)]
    out = dict(side=dict(id='p2', name='fp', pokemon=pokemon))
    if force:out['forceSwitch'] = [True]
    else:out['active'] = [dict(moves=moves)]
    return out


class OverlayTests(unittest.TestCase):
    def test_request_exact_hp_and_active_pp_replace_the_draws(self):
        state, notes = P.overlay_private(RT.STATE, 1, fp_request(), RT.VOCABULARY)
        self.assertEqual(notes, [])
        self.assertEqual(np.asarray(state.hp)[0, 1].tolist(), [100, 0, 73, 0, 1, 2])   # by species: Pikachu is slot 0, Snorlax slot 2
        self.assertEqual(np.asarray(state.pp)[0, 1, 2].tolist(), [3, 4, 2, 0])
        self.assertEqual(np.asarray(state.hp)[0, 0].tolist(), np.asarray(RT.STATE.hp)[0, 0].tolist())   # our side untouched
        self.assertEqual(np.asarray(RT.STATE.hp)[0, 1].tolist(), [100, 0, 50, 0, 1, 2])   # the world itself is not modified

    def test_what_the_request_cannot_prove_is_kept_and_noted(self):
        _, notes = P.overlay_private(RT.STATE, 1, dict(fp_request(), side=dict(fp_request()['side'], pokemon=[
            dict(details='Pikachu, L80', condition='90/120', active=False), dict(details='Mew, L80', condition='1/1')])), RT.VOCABULARY)
        self.assertIn('pikachu: max_hp 120 != world 100', notes);self.assertIn('mew: 0 world slots', notes)
        locked, notes = P.overlay_private(RT.STATE, 1, fp_request(active=[dict(id='thunderbolt', pp=1)]), RT.VOCABULARY)
        self.assertEqual(np.asarray(locked.pp)[0, 1, 2].tolist(), [5, 5, 5, 0]);self.assertIn('PP kept', notes[0])
        forced, notes = P.overlay_private(RT.STATE, 1, fp_request(force=True), RT.VOCABULARY)
        self.assertEqual((np.asarray(forced.pp)[0, 1, 2].tolist(), np.asarray(forced.hp)[0, 1, 2], notes), ([5, 5, 5, 0], 73, []))


class ViewTests(unittest.TestCase):
    def test_the_view_is_the_peer_inputs_prefix_and_the_peer_legal_lanes(self):
        class Executor:   # _make_views calls legal() then observe() per seat; each seat's inputs are told apart
            def legal(self, state):return jnp.asarray([[[True] * 9, [True, False, True, False, True, True, False, False, False]]])

            def observe(self, state, public, seat, mask):
                s = int(np.asarray(seat)[0])
                inputs = dict(numeric_features=jnp.full((1, 1, 29, 132), s, jnp.float32).at[0, 0, 22, 3].set(7 + s), categorical_ids=jnp.full((1, 1, 29, 41), 10 + s),
                    token_type_ids=jnp.asarray(np.r_[TOKEN_TYPES, [7] * 6])[None, None], attention_mask=jnp.ones((1, 1, 29), bool), history_mask=jnp.ones((1, 1), bool))
                return SimpleNamespace(inputs=inputs, observation=SimpleNamespace(legal_action_mask=mask, error=jnp.asarray([3 * s])))
        view = P.OpponentView(Executor())(RT.STATE, {}, 0)
        self.assertEqual((view['numeric_features'].shape, view['categorical_ids'].shape, view['token_type_ids'].shape), ((23, 132), (23, 41), (23,)))
        self.assertTrue((view['categorical_ids'] == 11).all());self.assertEqual((view['numeric_features'][22, 3], view['numeric_features'][0, 0]), (8., 1.))
        self.assertEqual(view['legal'].tolist(), [True, False, True, False, True, True, False, False, False]);self.assertEqual(view['error'], 3)
        self.assertEqual(view['token_type_ids'].tolist(), TOKEN_TYPES.tolist())
        two = SimpleNamespace(legal_action_mask=np.ones((2, 9), bool), error=np.zeros(2))
        with self.assertRaises(ValueError):P.view_arrays(two, dict(numeric_features=np.zeros((2, 29, 132)), categorical_ids=np.zeros((1, 29, 41)),
            token_type_ids=np.zeros((1, 29)), attention_mask=np.zeros((1, 29))))


# Env replay and alignment ---------------------------------------------------------------------------------------------------------
WAIT = '|request|' + json.dumps(dict(wait=True, side=dict(id='p1', name='a', pokemon=[BT.mon(i, i == 0) for i in range(6)])))
SCRIPT = (dict(turn=1, ours=False, fp=True, snorlax='73/100', pps=(3, 4, 2)), dict(turn=1, ours=None, fp=True, snorlax='73/100', pps=(3, 4, 2)),
    dict(turn=2, ours=False, fp=True, snorlax='60/100', pps=(2, 4, 2)), dict(turn=2, ours=False, fp=False, snorlax='60/100', pps=(2, 4, 2)),
    dict(turn=3, ours=True, fp=True, snorlax='60/100', pps=(2, 4, 1)), dict(turn=4, ours=False, fp=True, snorlax='41/100', pps=(1, 4, 1)))
# rounds 0..5: we decide (9 legal), we wait while Foul Play replaces, both, our replacement while Foul Play waits, our single-legal
# decision, both. Our decisions 0..4 sit in rounds 0, 2, 3, 4, 5 (request_count 1, 3, 4, 5, 6); positions: decisions 0, 1 and 4.
TEAM = [dict(species=n, details=f'{n}, L80') for n in ('Snorlax', 'Pikachu', 'Mr. Mime', "Farfetch’d", 'Ho-Oh', 'Gengar')]


def fake_view(state, public, owner):
    """A view determined by the foe's HP, its active member's PP and the world's public marker."""
    foe = 1 - owner;hp = np.asarray(state.hp)[0, foe];pp = np.asarray(state.pp)[0, foe, int(np.asarray(state.active)[0, foe])]
    numeric = np.zeros((23, 132), np.float32);numeric[1:7, 0] = hp / 100;numeric[13:17, 31] = pp / 10
    categorical = np.zeros((23, 41), np.int32);categorical[0, 0] = int(np.asarray(public['w']))
    return dict(numeric_features=numeric, categorical_ids=categorical, token_type_ids=TOKEN_TYPES.copy(), attention_mask=np.ones(23, bool),
        legal=np.asarray(RT.KEPT, bool), error=0)


def expected_row(step):
    """Foul Play's env observation in a round: what the true world's view is once its request-exact HP and PP are written."""
    view = fake_view(P.overlay_private(RT.STATE, 1, fp_request(step['snorlax'], step['pps']), RT.VOCABULARY)[0], dict(w=99), 0)
    return dict(numeric_features=view['numeric_features'], categorical_ids=view['categorical_ids'], token_type_ids=view['token_type_ids'],
        attention_mask=view['attention_mask'], legal=view['legal'])


class FakeEnv:
    """LocalShowdownEnv's replay surface over SCRIPT: requested players, literal requests, Foul Play's observations."""
    def __init__(self, script=SCRIPT, *, rows=None):
        self.script, self.rows, self.steps, self.closed, self.seeds, self.observed = script, rows, [], 0, [], []

    def reset(self, *, seed):
        self.seeds.append(seed);self.round = 0;self._load()

    def _load(self):
        if self.round >= len(self.script):self._latest_requests = {};return
        step = self.script[self.round];self._latest_turn = step['turn']
        ours = WAIT if step['ours'] is None else BT.own_request(step['ours'])
        self._latest_requests = dict(p1=json.loads(ours[len('|request|'):]), p2=fp_request(step['snorlax'], step['pps']) if step['fp'] else dict(wait=True))

    def terminal(self):
        return SimpleNamespace(winner='p1') if self.round >= len(self.script) else None

    def requested_players(self):
        step = self.script[self.round]
        return tuple(p for p, on in (('p1', step['ours'] is not None), ('p2', step['fp'])) if on)

    def observe(self, player):
        self.observed.append((self.round, player))
        if player == 'p1':return SimpleNamespace(legal_action_mask=(True,) * 9, metadata={})   # our view: observed, never kept
        row = (self.rows or {}).get(self.round) or expected_row(self.script[self.round])
        return SimpleNamespace(numeric_features=row['numeric_features'], categorical_ids=row['categorical_ids'], token_type_ids=row['token_type_ids'],
            attention_mask=row['attention_mask'], legal_action_mask=tuple(map(bool, row['legal'])), metadata=dict(self_team=TEAM))

    def step(self, actions):
        assert set(actions) == set(self.requested_players()), actions
        self.steps.append(dict(actions));self.round += 1;self._load()

    def close(self):
        self.closed += 1


def write_capture(path, script=SCRIPT, *, owner='p1', battle_id='battle-1001', stream_id=5):
    """A valid owner capture (phase_d_owning_capture's chain) of SCRIPT: one WIRE per round (a |turn| line when the turn moves on,
    then our request or wait request), a ROOT per round we decide."""
    blobs, state, prefix = [], dict(prior='0' * 64, lines=0, requests=0, decisions=0, turn=0), hashlib.sha256()

    def emit(**fields):
        blob = owning.encoded(dict(schema=owning.SCHEMA, sequence=len(blobs), previous_sha256=state['prior'], **fields))
        blobs.append(blob);state['prior'] = hashlib.sha256(blob).hexdigest()
    emit(event='BEGIN', cell=dict(ordinal=1001), declaration_sha256='d' * 64, namespace='ns', battle_id=battle_id, owner=owner, seed=7, stream_id=stream_id)
    for generation, step in enumerate(script, 1):
        line = WAIT if step['ours'] is None else BT.own_request(step['ours'])
        delta = ([f'|turn|{step["turn"]}'] if step['turn'] != state['turn'] else []) + [line];state['turn'] = step['turn']
        for x in delta:prefix.update(x.encode() + b'\n')
        emit(event='WIRE', generation=generation, start=state['lines'], end=state['lines'] + len(delta), prefix_sha256=prefix.hexdigest(), lines=delta)
        state['lines'] += len(delta);state['requests'] += 1
        if step['ours'] is not None:
            emit(event='ROOT', decision_index=state['decisions'], turn=step['turn'], generation=generation, request_count=state['requests'],
                prefix_sha256=prefix.hexdigest(), own_request_line=line)
            state['decisions'] += 1
    path.write_bytes(b''.join(b + b'\n' for b in blobs))


def fp_rows(script=SCRIPT, *, seed=7, seat=0):
    """fp_reply_data's rows of the game: one per round Foul Play is requested (its env observation, label 0)."""
    return {r: dict(expected_row(step), turn=np.int64(step['turn']), forced=np.bool_(False), label=np.int64(0), seed=np.int64(seed), seat=np.int64(seat))
        for r, step in enumerate(script) if step['fp']}


def game(script=SCRIPT):
    journal = {r: dict(round=r, seat='p2', choice='move 1', action=0) for r, step in enumerate(script) if step['fp']}
    return dict(ordinal=1001, seed=7, seat=0, fp='p2', ours='p1', journal=journal, our_actions=[0] * sum(s['ours'] is not None for s in script))


def captured(tmp):
    path = Path(tmp) / 'c.jsonl';write_capture(path)
    return list(owning.read_capture(path))


class ReplayTests(unittest.TestCase):
    def test_rounds_align_through_request_count(self):
        with tempfile.TemporaryDirectory() as tmp:rows = captured(tmp)
        env = FakeEnv();table, team = P.replay_rounds(env, game(), {r['decision_index']: r['own_request_line'] for r in rows})
        self.assertEqual(env.seeds, [7]);self.assertEqual(team, TEAM);self.assertEqual(len(env.steps), 6)
        self.assertEqual(env.observed, [(0, 'p1'), (0, 'p2'), (1, 'p2'), (2, 'p1'), (2, 'p2'), (3, 'p1'), (4, 'p1'), (4, 'p2'), (5, 'p1'), (5, 'p2')])
        self.assertEqual([table[r]['our_decision'] for r in range(6)], [0, None, 1, 2, 3, 4])
        self.assertEqual([table[r]['fp_action'] for r in range(6)], [0, 0, 0, None, 0, 0])
        self.assertEqual(table[2]['fp_request']['side']['pokemon'][0]['condition'], '60/100')
        self.assertEqual([r['request_count'] for r in rows], [1, 3, 4, 5, 6])
        aligned = [P.align(r, table, fp_rows()) for r in rows]
        self.assertEqual([a['round'] for a in aligned], [0, 2, 3, 4, 5])
        self.assertEqual([a['aligned'] for a in aligned], [True, True, False, True, True])
        self.assertEqual(aligned[2]['reason'], 'fp_not_requested')
        self.assertTrue(all(a['turns_equal'] and all(a['env_equals_fp_reply'].values()) and a['label_equal'] for a in aligned if a['aligned']))
        self.assertEqual(aligned[1]['turns'], dict(capture=2, env=2, fp_reply=2))
        rows_off = fp_rows();rows_off[2] = dict(rows_off[2], turn=np.int64(9), label=np.int64(4));rows_off[2]['numeric_features'] = rows_off[2]['numeric_features'] + 1
        off = P.align(rows[1], table, rows_off)
        self.assertTrue(off['aligned']);self.assertFalse(off['turns_equal']);self.assertFalse(off['label_equal'])
        self.assertEqual(off['env_equals_fp_reply'], dict(numeric_features=False, categorical_ids=True, token_type_ids=True, attention_mask=True, legal=True))
        missing = fp_rows();del missing[5]
        self.assertEqual(P.align(rows[4], table, missing)['reason'], 'no_fp_reply_row')
        shifted = dict(rows[1], request_count=2)   # a round that is not this decision's
        self.assertEqual(P.align(shifted, table, fp_rows())['reason'], 'our_decision_elsewhere')
        self.assertEqual(P.align(dict(rows[1], request_count=40), table, fp_rows())['reason'], 'round_not_replayed')

    def test_the_replay_refuses_a_game_it_cannot_reproduce(self):
        with tempfile.TemporaryDirectory() as tmp:rows = captured(tmp)
        requests = {r['decision_index']: r['own_request_line'] for r in rows}
        with self.assertRaisesRegex(P.Mismatch, 'differs from the capture at decision 1'):
            P.replay_rounds(FakeEnv(), game(), {**requests, 1: BT.own_request(True)})
        g = game();del g['journal'][2]
        with self.assertRaisesRegex(P.Mismatch, 'round 2 without a journal entry'):P.replay_rounds(FakeEnv(), g, requests)
        g = game();g['journal'][3] = dict(round=3, seat='p2', action=0)
        with self.assertRaisesRegex(P.Mismatch, 'never requested'):P.replay_rounds(FakeEnv(), g, requests)
        g = game();g['journal'][0] = dict(round=0, seat='p2', action=1)   # lane 1 is not legal in KEPT
        with self.assertRaisesRegex(P.Mismatch, 'illegal'):P.replay_rounds(FakeEnv(), g, requests)
        g = game();g['our_actions'] = g['our_actions'] + [0]
        with self.assertRaisesRegex(P.Mismatch, 'made 5 of 6'):P.replay_rounds(FakeEnv(), g, requests)

    def test_request_bodies_ignore_rqid_and_the_username(self):
        a = '|request|' + json.dumps(dict(rqid=3, side=dict(id='p1', name='x', pokemon=[1])))
        self.assertEqual(P.request_body(a), P.request_body(json.dumps(dict(rqid=9, side=dict(id='p1', name='y', pokemon=[1])))))
        self.assertNotEqual(P.request_body(a), P.request_body(json.dumps(dict(side=dict(id='p1', pokemon=[2])))))


# Cells, games and reply rows -----------------------------------------------------------------------------------------------------
def write_result(root, ordinal, *, arm='RAW', seed=7, seat=0, status='COMPLETE', fp='p2', journal=None, capture=True):
    for d in ('results', 'owning-wire', 'telemetry'):(root / d).mkdir(parents=True, exist_ok=True)
    journal = [dict(round=0, seat=fp, choice='move 1', action=0)] if journal is None else journal
    (root / 'results' / f'{ordinal}.json').write_text(json.dumps(dict(status=status, cell=dict(ordinal=ordinal, arm=arm, seed=seed, seat=seat, pair_ordinal=ordinal),
        source_game=dict(seed=seed, opponent_moves=journal), benchmark=dict(foulplay_player=fp))))
    if capture:(root / 'owning-wire' / f'{ordinal}.jsonl').write_text('')


def write_shard(path, rows, document=None):
    """An fp_reply_data shard: rows of (ordinal, round, value)."""
    n = len(rows)
    arrays = dict(numeric_features=np.stack([np.full((23, 132), v, np.float32) for _, _, v in rows]), categorical_ids=np.zeros((n, 23, 41), np.int32),
        token_type_ids=np.zeros((n, 23), np.int16), attention_mask=np.ones((n, 23), bool), legal=np.ones((n, 9), bool), probs=np.zeros((n, 9), np.float32),
        value=np.zeros(n, np.float32), label=np.zeros(n, np.int64), round=np.asarray([r for _, r, _ in rows]), turn=np.ones(n, np.int64), forced=np.zeros(n, bool),
        ordinal=np.asarray([o for o, _, _ in rows]), seed=np.full(n, 7), seat=np.zeros(n, np.int64))
    np.savez(path, **arrays)
    if document is not None:path.with_suffix('.json').write_text(json.dumps(document))


class CellTests(unittest.TestCase):
    def test_load_game_reads_seats_journal_and_our_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);write_result(root, 3, journal=[dict(round=0, seat='p2', action=1), dict(round=2, seat='p2', action=4)])
            (root / 'telemetry' / '3.jsonl').write_text(''.join(json.dumps(dict(owner=o, decision_index=i, played_action_index=a)) + '\n'
                for o, i, a in (('p1', 1, 5), ('p1', 0, 2), ('p2', 0, 9))))
            g = P.load_game(root, dict(ordinal=3, seed=7, seat=0))
            self.assertEqual((g['fp'], g['ours'], g['seed'], g['our_actions'], sorted(g['journal'])), ('p2', 'p1', 7, [2, 5], [0, 2]))
            for kw, kind in ((dict(seed=2103400017), 'screen_band'), (dict(status='UNKNOWN'), 'not_complete'), (dict(journal=[]), 'no_journal'),
                    (dict(fp=None), 'unknown_seats'), (dict(journal=[dict(round=0, seat='p2', action=0)] * 2), 'journal_shape')):
                write_result(root, 4, **kw)
                with self.subTest(kind=kind), self.assertRaises(P.CellError) as caught:P.load_game(root, dict(ordinal=4, seed=kw.get('seed', 7)))
                self.assertEqual(caught.exception.kind, kind)
            (root / 'telemetry' / '3.jsonl').write_text(json.dumps(dict(owner='p1', decision_index=1, played_action_index=0)) + '\n')
            with self.assertRaises(P.CellError) as caught:P.load_game(root, dict(ordinal=3, seed=7))
            self.assertEqual(caught.exception.kind, 'telemetry_gap')
            with self.assertRaises(P.CellError) as caught:P.load_game(root, dict(ordinal=3, seed=8))
            self.assertEqual(caught.exception.kind, 'seed_differs')

    def test_cells_are_replayable_raw_cells_outside_the_screen_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for o in range(40):write_result(root, o, seed=100 + o // 2, seat=o % 2)
            write_result(root, 40, arm='DN2W16');write_result(root, 41, seed=2103400005);write_result(root, 42, capture=False);write_result(root, 43, status='UNKNOWN')
            write_result(root, 44)   # no reply rows
            fp = frozenset(range(44))
            args = lambda **kw: SimpleNamespace(**dict(dict(collection=str(root), cells=None, sample=None, sample_seed=None, shard=0, shards=1), **kw))
            self.assertEqual([c['ordinal'] for c in P.select_cells(args(cells=[5, 2, 5]), fp)], [2, 5])
            for bad, why in ((40, 'not_raw'), (41, 'screen_band'), (42, 'no_owner_capture'), (43, 'not_complete'), (44, 'no_fp_reply_rows')):
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, why):P.select_cells(args(cells=[1, bad]), fp)
            picked = P.select_cells(args(sample=8, sample_seed=1), fp)
            self.assertEqual(len(picked), 8);self.assertTrue(all(c['ordinal'] < 40 for c in picked))
            self.assertEqual(picked, P.select_cells(args(sample=8, sample_seed=1), fp))
            shards = [P.select_cells(args(sample=8, sample_seed=1, shard=i, shards=2), fp) for i in range(2)]
            self.assertEqual(sorted(c['ordinal'] for s in shards for c in s), [c['ordinal'] for c in picked])

    def test_reply_rows_index_load_and_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_shard(d / 'shard-0.npz', [(4, 0, 1.), (4, 2, 2.), (6, 0, 3.)], dict(schema='v2s-fp-reply-data.v1', games=2, decisions=3, ignored=1))
            write_shard(d / 'shard-1.npz', [(5, 1, 4.)])
            rows = P.FpReplyRows(d)
            self.assertEqual(rows.ordinals, {4, 5, 6});self.assertEqual(rows.documents, {'shard-0.json': dict(schema='v2s-fp-reply-data.v1', games=2, decisions=3,
                mismatched=None, checkpoint_sha256=None, shapes=None, encoder_parity=None)})
            loaded = rows.load([4, 5])
            self.assertEqual(sorted(loaded), [4, 5]);self.assertEqual(sorted(loaded[4]), [0, 2])
            self.assertEqual(float(loaded[4][2]['numeric_features'][0, 0]), 2.);self.assertEqual(loaded[5][1]['legal'].shape, (9,))
            self.assertIsNone(loaded[4][0]['numeric_features'].base)   # a copy: no shard array stays alive
            write_shard(d / 'shard-2.npz', [(6, 0, 5.)])
            with self.assertRaisesRegex(ValueError, 'twice'):P.FpReplyRows(d)
            np.savez(d / 'shard-2.npz', ordinal=np.zeros(1), round=np.zeros(1))
            with self.assertRaisesRegex(ValueError, 'missing'):P.FpReplyRows(d)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'no shard'):P.FpReplyRows(tmp)


# The sampler wrapper ---------------------------------------------------------------------------------------------------------------
SAMPLED = []


def fake_sample(root, detached, *, lines, catalog, source, metadata, config, battle_id, seed, count):
    """The constructor's sampler stand-in: RT's 16 worlds (world i's public marker i, kernel seed 1000 + i) for its own root."""
    SAMPLED.append(detached.address.frame)
    if detached.address.frame == 99:raise Refusal('SAME_PUBLIC_ROOT_SOURCE_BINDING_REQUIRED')
    return SimpleNamespace(root=detached, worlds=tuple(RT.world(i) for i in range(16)), seeds=tuple(500 + i for i in range(16)), requested_worlds=16,
        fully_constructed=True, mapper_diagnostics=())


ROOT_CARRY = SimpleNamespace(kind='the root\'s own public carry')   # what the completion receives as `root`


class SamplingConstructor(RT.FakeConstructor):
    """RT's constructor, its prepare sampling through phase_d_protocol_world.sample as ProtocolConstructor.prepare does."""
    owning_config = 'owning-config'

    def prepare(self, request, index):
        self.calls.append(request['decision_index'])
        if request['decision_index'] in self.refuse:raise Refusal('TEST_REFUSAL', 'no worlds')
        detached = SimpleNamespace(address=SimpleNamespace(stream=1, frame=request['decision_index'], owner=0), public=ROOT_CARRY)
        made = pw.sample(SimpleNamespace(kind='public'), detached, lines=tuple(request['lines']), catalog='catalog', source='source', metadata='metadata',
            config='config', battle_id=request['battle_id'], seed=20261103, count=16)
        return made.root, made, RT.LEGAL


class WrapperTests(unittest.TestCase):
    def test_the_call_site_resolves_through_the_module_at_call_time(self):
        self.assertTrue(P.check_call_site())

        def bound(self, request, index):   # a constructor holding its own reference to sample would escape the wrapper
            sample = pw.sample
            return sample(None, None)
        with self.assertRaises(RuntimeError):P.check_call_site(bound)

    def test_the_wrapper_captures_restores_and_leaves_responses_identical(self):
        requests = [RT.body(0), RT.body(1), RT.body(2), RT.body(99)]
        with ExitStack() as stack:
            RT.pricing(stack, still_clock=True);stack.enter_context(patch.object(pw, 'sample', fake_sample))
            plain = RT.fake_search(SamplingConstructor(refuse={1}))
            without = [plain.process_many([r])[0] for r in requests]
            wrapped, captures, within = RT.fake_search(SamplingConstructor(refuse={1})), [], []
            for r in requests:
                capture = P.SampleCapture()
                with capture.installed():
                    self.assertIsNot(pw.sample, fake_sample);within.append(wrapped.process_many([r])[0])
                self.assertIs(pw.sample, fake_sample);captures.append(capture)
        self.assertIs(pw.sample, ORIGINAL_SAMPLE)
        self.assertEqual([r['status'] for r in without], ['COMPLETE', 'REFUSED', 'COMPLETE', 'REFUSED'])
        self.assertEqual(json.dumps(within, sort_keys=True), json.dumps(without, sort_keys=True))   # byte for byte, timings included
        self.assertEqual([len(c.calls) for c in captures], [1, 0, 1, 1])
        first = captures[0].calls[0]
        self.assertEqual((first.named['detached'].address.frame, first.named['lines'], first.named['seed'], first.named['count'], first.named['battle_id']),
            (0, tuple(RT.body(0)['lines']), 20261103, 16, 'b0'))
        self.assertEqual([w.kernel_seed for w in first.result.worlds], [1000 + i for i in range(16)]);self.assertIsNone(first.error)
        self.assertTrue(captures[3].calls[0].error.startswith('Refusal: SAME_PUBLIC_ROOT_SOURCE_BINDING_REQUIRED'))
        self.assertIsNone(captures[3].calls[0].result);self.assertEqual(without[3]['cause'], 'constructor_refused')

    def test_the_wrapper_restores_when_the_call_raises(self):
        with self.assertRaises(KeyError):
            with P.SampleCapture().installed():raise KeyError('boom')
        self.assertIs(pw.sample, ORIGINAL_SAMPLE)


# Capture of one cell, end to end on fakes --------------------------------------------------------------------------------------------
def write_cell(root):
    journal = [dict(round=r, seat='p2', choice='move 1', action=0) for r, step in enumerate(SCRIPT) if step['fp']]
    write_result(root, 1001, journal=journal)
    write_capture(root / 'owning-wire' / '1001.jsonl')
    (root / 'telemetry' / '1001.jsonl').write_text(''.join(json.dumps(dict(owner='p1', decision_index=i, played_action_index=0)) + '\n' for i in range(5)))


def fake_true_world(seen):
    """build_true_world's stand-in: RT's world (Snorlax at 50 HP, PP 5/5/5) under public marker 99, world 0's kernel seed."""
    def build(call, team, request0, *, index=0):
        seen.append((call.named['detached'].address.frame, team, request0))
        return SimpleNamespace(state=RT.STATE, public=dict(w=jnp.asarray(99))), dict(reason=None, kernel_seed=1000, constructed=True, refusals=[], notes=[])
    return build


class CaptureTests(unittest.TestCase):
    def run_cell(self, tmp, stack, *, env=None, complete=None):
        root = Path(tmp) / 'run';write_cell(root);out = Path(tmp) / 'out';out.mkdir()
        RT.pricing(stack);stack.enter_context(patch.object(pw, 'sample', fake_sample))
        seen = [];stack.enter_context(patch.object(P, 'build_true_world', fake_true_world(seen)))
        constructor = SamplingConstructor();env = FakeEnv() if env is None else env
        ctx = P.Replay(run=root, search=RT.fake_search(constructor), views=fake_view, fp_rows={1001: fp_rows()}, vocabulary=RT.VOCABULARY,
            environment_factory=lambda: env, sampled_worlds=2, shard=0, out=out, complete=complete)
        with (out / 'positions-0.jsonl').open('x') as stream, patch('sys.stdout', new_callable=io.StringIO) as stdout:
            ctx.stream = stream;count = P.capture_cell(ctx, P._rr().cell_of(root, 1001))
        return ctx, count, constructor, seen, out, stdout.getvalue()

    def test_one_cell_is_replayed_in_decision_order_and_captured(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            ctx, count, constructor, seen, out, stdout = self.run_cell(tmp, stack)
            rows = P.read_rows(out / 'positions-0.jsonl')
            with np.load(out / 'arrays-0-1001.npz') as data:arrays = {k: np.asarray(data[k]) for k in data.files}
            report = P.build_report([out])
        self.assertIs(pw.sample, ORIGINAL_SAMPLE)
        self.assertEqual(count, 3);self.assertEqual(constructor.calls, [0, 1, 2, 4])   # every sent decision, in order; decision 3 is single-legal
        self.assertEqual(dict(ctx.counts), dict(sent=4, skipped_single_legal=1, positions=3, **{'not_aligned:fp_not_requested': 1}))
        self.assertEqual([(r['decision_index'], r['round'], r['request_count']) for r in rows], [(0, 0, 1), (1, 2, 3), (4, 5, 6)])
        self.assertEqual([s[0] for s in seen], [0, 1, 4]);self.assertEqual(seen[0][1], TEAM)
        self.assertEqual(seen[0][2]['side']['pokemon'][0]['condition'], '73/100')   # the true team's literal request is round 0's, at every position
        self.assertEqual(len(stdout.splitlines()), 3)
        first = rows[0]
        self.assertEqual((first['response']['status'], first['fp'], first['ours'], first['npz'], first['row']), ('COMPLETE', 'p2', 'p1', 'arrays-0-1001.npz', 0))
        self.assertEqual(first['alignment']['turns'], dict(capture=1, env=1, fp_reply=1));self.assertTrue(first['worlds']['true']['kernel_seed_is_world0s'])
        self.assertEqual(first['worlds']['sampled_1'], dict(index=1, kernel_seed=1001, refusals=[], constructed=True));self.assertEqual(first['notes'], [])
        for r in rows:
            self.assertTrue(r['exact']['true_private']['all'])   # Foul Play's observation is the true world with its exact HP and PP
            self.assertEqual((r['exact']['true']['numeric_features'], r['exact']['true']['categorical_ids']), (False, True))
            self.assertEqual((r['exact']['sampled_0']['categorical_ids'], r['exact']['sampled_0']['legal']), (False, True))
        self.assertEqual(arrays['view_numeric'].shape, (3, 4, 23, 132));self.assertEqual(arrays['kinds'].tolist(), ['true', 'true_private', 'sampled_0', 'sampled_1'])
        self.assertEqual(arrays['round'].tolist(), [0, 2, 5]);self.assertTrue(arrays['view_valid'].all())
        self.assertTrue(np.array_equal(arrays['env_numeric'][1], expected_row(SCRIPT[2])['numeric_features']))
        # The report over the capture: the overlay closes every gap; the true world differs in Foul Play's own HP and its active PP only.
        self.assertEqual(report['positions'], 3);self.assertEqual(report['parity']['true_private']['exact']['observation_and_legal'], 1.)
        true = report['parity']['true']
        self.assertEqual((true['exact']['numeric'], true['exact']['categorical'], true['exact']['legal']), (0., 1., 1.))
        self.assertEqual({(c['token'], c['column']) for c in true['numeric_cells']}, {(3, 0), (13, 31), (14, 31), (15, 31)})
        self.assertEqual({c['token_label'] for c in true['numeric_cells']}, {'self2', 'move0', 'move1', 'move2'})
        self.assertEqual(true['numeric_cells'][0]['examples'][0]['view'], .5)
        self.assertEqual(report['parity']['sampled_1']['categorical_cells'][0]['column_name'], 'PRIMARY')
        self.assertEqual(report['sampled_vs_true']['sampled_0']['exact']['numeric'], 1.)
        self.assertEqual(report['alignment']['env_equals_fp_reply_all'], 1.);self.assertEqual(report['worlds']['true']['kernel_seed_is_world0s'], {'True': 3})

    def test_every_world_kind_is_completed_with_its_decisions_lines_before_its_view(self):
        calls = []

        def complete(state, public, owner, lines, root=None):   # marks both: categorical [0, 0] reads the carry, numeric self0 HP the state
            calls.append((int(np.asarray(public['w'])), owner, lines, root))
            if int(np.asarray(public['w'])) == 1:raise ValueError('PEER_REQUEST_ORDER_UNAVAILABLE stand-in')
            return state._replace(hp=state.hp.at[0, 1, 0].set(7)), dict(w=jnp.asarray(int(np.asarray(public['w'])) + 200))
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            ctx, count, constructor, seen, out, stdout = self.run_cell(tmp, stack, complete=complete)
            rows = P.read_rows(out / 'positions-0.jsonl')
            with np.load(out / 'arrays-0-1001.npz') as data:arrays = {k: np.asarray(data[k]) for k in data.files}
        self.assertEqual(count, 3);self.assertEqual(len(calls), 3 * 4)   # true, true_private, sampled_0, sampled_1 per position
        self.assertEqual([c[0] for c in calls[:4]], [99, 99, 0, 1]);self.assertTrue(all(c[1] == 0 for c in calls))
        self.assertTrue(all(c[3] is ROOT_CARRY for c in calls))   # the root's own carry (the Pressure count's base), for every kind
        first = rows[0];lines = calls[0][2]
        self.assertTrue(lines and lines[-1].startswith('|request|'))   # the decision's own prefix, its last line the own request
        self.assertEqual(arrays['view_categorical'][0, :3, 0, 0].tolist(), [299, 299, 200])   # completed carries; sampled_1's raised
        np.testing.assert_allclose(arrays['view_numeric'][0, :3, 1, 0], [.07] * 3)   # the views read the completed STATE too
        self.assertFalse(arrays['view_valid'][0, 3]);self.assertTrue(any('sampled_1 view' in n and 'PEER_REQUEST_ORDER' in n for n in first['notes']))

    def test_a_game_the_env_cannot_replay_is_a_cell_error_before_anything_is_sent(self):
        class Broken(FakeEnv):
            def step(self, actions):raise RuntimeError('bridge died')
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack, self.assertRaises(P.CellError) as caught:
            env = Broken();self.run_cell(tmp, stack, env=env)
        self.assertEqual(caught.exception.kind, 'env_error');self.assertEqual(env.closed, 1)

    def test_max_positions_stops_a_smoke_run(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp) / 'run';write_cell(root)
            RT.pricing(stack);stack.enter_context(patch.object(pw, 'sample', fake_sample));stack.enter_context(patch.object(P, 'build_true_world', fake_true_world([])))
            constructor = SamplingConstructor();out = Path(tmp) / 'out';out.mkdir()
            ctx = P.Replay(run=root, search=RT.fake_search(constructor), views=fake_view, fp_rows={1001: fp_rows()}, vocabulary=RT.VOCABULARY,
                environment_factory=FakeEnv, max_positions=2, out=out)
            with (out / 'positions-0.jsonl').open('x') as stream, patch('sys.stdout', new_callable=io.StringIO):
                ctx.stream = stream;P.capture_cell(ctx, P._rr().cell_of(root, 1001))
        self.assertEqual((ctx.positions, ctx.done, constructor.calls), (2, True, [0, 1]))   # stopped before the next sent decision


BASE = ['--showdown-root', 's', '--node-binary', 'n', '--metadata', 'm', '--expected-metadata-sha256', 'x', '--vocabulary', 'v', '--expected-vocabulary-sha256', 'y',
    '--warmup-requests', 'w.jsonl', '--rung', 'DN2W16', '--collection', 'r', '--fp-reply-data', 'f', '--out', 'o']


class MainTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'JAX_DEFAULT_MATMUL_PRECISION': 'highest'});environment.start();self.addCleanup(environment.stop)

    def test_options(self):
        with patch.object(P._rr().server.evaluators, 'check_arguments'):
            args = P.capture_options(BASE + RT.CHAMPION + ['--cells', '1001'])
            self.assertEqual((args.buckets, args.max_requests, args.saved_root_admission, args.cells, args.sampled_worlds, args.rung), ((1,), 1, False, [1001], 2, 'DN2W16'))
            for argv in (BASE + ['--cells', '1'], [a for a in BASE if a not in ('--rung', 'DN2W16')] + RT.CHAMPION + ['--cells', '1'],
                    BASE + RT.CHAMPION + ['--sample', '4'], BASE + RT.CHAMPION + ['--cells', '1', '--sample', '4', '--sample-seed', '1'],
                    BASE + RT.CHAMPION + ['--cells', '1', '--sampled-worlds', '17'], BASE + RT.CHAMPION + ['--cells', '1', '--buckets', '1,2'],
                    BASE + RT.CHAMPION + ['--cells', '1', '--max-positions', '0'], BASE + RT.CHAMPION + ['--cells', '1', '--sessions', '2']):
                with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):P.capture_options(argv)
            self.assertEqual(P.capture_options(BASE + RT.CHAMPION + ['--sample', '4', '--sample-seed', '3', '--shard', '1', '--shards', '2']).shard, 1)

    def capture(self, tmp, header=None):
        root = Path(tmp) / 'run';write_cell(root);out = Path(tmp) / 'out'
        args = P.capture_parser().parse_args(BASE + RT.CHAMPION + ['--cells', '1001']);args.collection, args.out, args.port, args.max_requests = str(root), str(out), 0, 1
        constructor = SamplingConstructor();search = RT.fake_search(constructor);warmed = []
        search.warmup = lambda requests: warmed.append(list(requests)) or dict(programs=3)
        fpr = SimpleNamespace(ordinals=frozenset({1001}), load=lambda ordinals: {1001: fp_rows()}, paths=[Path('shard-0.npz')], documents={'shard-0.json': {}})
        header = dict(P._rr().LIVE_IDENTITY, rung='DN2W16', pricer=dict(buckets=[1], W=16)) if header is None else header
        server = P._rr().server;envs = []
        with ExitStack() as stack:
            RT.pricing(stack);stack.enter_context(patch.object(pw, 'sample', fake_sample))
            stack.enter_context(patch.object(P, 'build_true_world', fake_true_world([])))
            stack.enter_context(patch.object(P, 'capture_options', return_value=args));stack.enter_context(patch.object(P, 'tree_state', return_value=dict(head='h')))
            stack.enter_context(patch.object(P, 'FpReplyRows', return_value=fpr));stack.enter_context(patch.object(P, 'OpponentView', return_value=fake_view))
            stack.enter_context(patch.object(P, 'make_environment', side_effect=lambda config: envs.append(config) or FakeEnv()))
            stack.enter_context(patch.object(P, 'make_completion', return_value=None))   # the fake worlds' carries are kept as built
            stack.enter_context(patch.object(server, 'read_requests', return_value=[dict(saved=1)]));stack.enter_context(patch.object(server, 'build', return_value=search))
            stack.enter_context(patch.object(server, 'header', return_value=header))
            with patch('sys.stdout', new_callable=io.StringIO):code = P.main(['capture'])
        return code, out, warmed, envs

    def test_capture_main_writes_header_positions_arrays_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, warmed, envs = self.capture(tmp)
            header, summary = P.read_json(out / 'header-0.json'), P.read_json(out / 'summary-0.json')
            positions = P.read_rows(out / 'positions-0.jsonl');files = sorted(p.name for p in out.iterdir())
            with patch('sys.stdout', new_callable=io.StringIO):self.assertEqual(P.main(['report', '--capture', str(out), '--out', str(Path(tmp) / 'report')]), 0)
            markdown = (Path(tmp) / 'report' / 'summary.md').read_text();report = P.read_json(Path(tmp) / 'report' / 'summary.json')
            with patch('sys.stdout', new_callable=io.StringIO), self.assertRaises(FileExistsError):
                P.main(['report', '--capture', str(out), '--out', str(Path(tmp) / 'report')])   # never over an earlier report
        self.assertEqual(code, 0);self.assertEqual(warmed, [[dict(saved=1)]]);self.assertEqual(envs, ['owning-config'])
        self.assertEqual(files, ['arrays-0-1001.npz', 'header-0.json', 'positions-0.jsonl', 'summary-0.json'])
        self.assertEqual((header['kinds'], header['tree'], header['cells'][0]['ordinal']), (['true', 'true_private', 'sampled_0', 'sampled_1'], dict(head='h'), 1001))
        self.assertIs(header['carry_completion'], False)
        self.assertEqual((summary['positions'], summary['sent'], summary['skipped_single_legal'], summary['not_aligned'], summary['cell_errors'], summary['interrupted'],
            summary['cells_replayed']), (3, 4, 1, dict(fp_not_requested=1), [], False, 1))
        self.assertEqual(len(positions), 3);self.assertEqual(report['capture']['not_aligned'], dict(fp_not_requested=1))
        self.assertIn('| true_private | 3 | 1.000 |', markdown);self.assertIn('self2 (3) | 0 | HP_FRACTION', markdown)

    def test_a_search_unlike_the_live_servers_is_refused_before_warmup(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(SystemExit, 'session_cap'):
            self.capture(tmp, header=dict(P._rr().LIVE_IDENTITY, rung='DN2W16', session_cap=64, pricer=dict(buckets=[1], W=16)))


# Report math ----------------------------------------------------------------------------------------------------------------------
def toy(n=5):
    env = dict(numeric=np.zeros((n, 23, 132), np.float32), categorical=np.zeros((n, 23, 41), np.int32), token_type=np.tile(TOKEN_TYPES, (n, 1)),
        attention=np.ones((n, 23), bool), legal=np.ones((n, 9), bool))
    return env, {k: v.copy() for k, v in env.items()}


def synthetic_capture(directory, env, view, recomputed=None):
    """A capture directory of one world kind ('true') whose env and view numeric features are `env` / `view` [n, 23, 132] (everything
    else equal), with env_opp_hazard_credit `recomputed` [n] when given (an earlier capture's layout otherwise)."""
    directory.mkdir();n = len(env);t = np.tile(TOKEN_TYPES, (n, 1)).astype(np.int16)
    extra = {} if recomputed is None else dict(env_opp_hazard_credit=np.asarray(recomputed, np.float32))
    np.savez(directory / 'arrays-0-1.npz', kinds=np.asarray(['true']), env_numeric=env, env_categorical=np.zeros((n, 23, 41), np.int32), env_token_type=t,
        env_attention=np.ones((n, 23), bool), env_legal=np.ones((n, 9), bool), view_numeric=view[:, None], view_categorical=np.zeros((n, 1, 23, 41), np.int32),
        view_token_type=t[:, None], view_attention=np.ones((n, 1, 23), bool), view_legal=np.ones((n, 1, 9), bool), view_valid=np.ones((n, 1), bool),
        view_error=np.zeros((n, 1), np.int32), ordinal=np.ones(n, np.int64), round=np.arange(n), decision_index=np.arange(n), turn=np.ones(n, np.int64), **extra)
    (directory / 'positions-0.jsonl').write_text(''.join(json.dumps(dict(views=dict(true={}))) + '\n' for _ in range(n)))
    (directory / 'summary-0.json').write_text(json.dumps(dict(positions=n)))
    return directory


class ReportMathTests(unittest.TestCase):
    def test_shares_maxima_and_ranking(self):
        env, view = toy()
        view['numeric'][0, 1, 0] = .5;view['numeric'][1, 1, 0] = 1e-7   # one real gap, one below tolerance, same cell
        view['numeric'][0, 5, 20] = -.25   # a second cell, the first position
        view['categorical'][2, 7, 3] = 9;view['attention'][3, 22] = False;view['legal'][3, 4] = False;view['legal'][4, 0] = True;env['legal'][4, 0] = False
        s = P.diff_stats(env, view, tolerance=1e-5)
        self.assertEqual(s['positions'], 5)
        self.assertEqual(s['exact'], dict(numeric=.6, numeric_within_tolerance=.8, categorical=.8, token_type=1., attention=.8, legal=.6, observation=.2,
            observation_and_legal=0.))
        self.assertAlmostEqual(float(s['numeric_cell_share'][1, 0]), .4);self.assertAlmostEqual(float(s['numeric_cell_beyond'][1, 0]), .2)
        self.assertAlmostEqual(float(s['numeric_cell_max'][1, 0]), .5);self.assertAlmostEqual(float(s['numeric_column_max'][20]), .25)
        self.assertAlmostEqual(float(s['token_numeric_share'][1]), .4);self.assertAlmostEqual(float(s['token_categorical_share'][7]), .2)
        self.assertAlmostEqual(float(s['attention_share'][22]), .2);self.assertAlmostEqual(float(s['categorical_column_share'][3]), .2)
        self.assertEqual(s['legal_lane_env_only'].tolist(), [0, 0, 0, 0, 1, 0, 0, 0, 0]);self.assertEqual(s['legal_lane_view_only'].tolist(), [1] + [0] * 8)
        names, labels = P.numeric_names(), P.token_labels()
        ranked = P.ranked_cells(s, 'numeric', labels, names)
        self.assertEqual([(c['token'], c['column']) for c in ranked], [(1, 0), (5, 20)])   # equal beyond-tolerance shares: the larger exact share first
        self.assertEqual((ranked[0]['share'], ranked[0]['beyond_tolerance_share'], ranked[0]['positions'], ranked[0]['token_label'], ranked[0]['column_name']),
            (.4, .2, 2, 'self0', 'HP_FRACTION'))
        keys = dict(ordinal=np.arange(10, 15), round=np.arange(5), decision_index=np.arange(5) * 2, turn=np.ones(5, int))
        examples = P.cell_examples(env, view, keys, 'numeric', 1, 0, 5)
        self.assertEqual([(e['ordinal'], e['env'], e['view']) for e in examples], [(10, 0., .5), (11, 0., float(np.float32(1e-7)))])   # the largest gap first
        decode = lambda i: f'tok{i}'
        self.assertEqual(P.cell_examples(env, view, keys, 'categorical', 7, 3, 5, decode), [dict(ordinal=12, round=2, decision_index=4, turn=1, env=0, view=9,
            env_token='tok0', view_token='tok9')])
        self.assertEqual([(c['token'], c['column'], c['column_name']) for c in P.ranked_cells(s, 'categorical', labels, P.categorical_names())], [(7, 3, 'SLOT')])

    def test_ranking_puts_real_gaps_before_float_noise(self):
        env, view = toy(4)
        view['numeric'][:, 2, 9] += 1e-7   # every position, below tolerance
        view['numeric'][0, 3, 4] = 1.   # one position, a real gap
        ranked = P.ranked_cells(P.diff_stats(env, view, tolerance=1e-5), 'numeric', P.token_labels(), P.numeric_names())
        self.assertEqual([(c['token'], c['column'], c['share']) for c in ranked], [(3, 4, .25), (2, 9, 1.)])

    def test_nan_on_both_sides_is_equal_on_one_side_a_gap(self):
        env, view = toy(2)
        env['numeric'][:, 0, 0] = np.nan;view['numeric'][0, 0, 0] = np.nan
        s = P.diff_stats(env, view, tolerance=1e-5)
        self.assertEqual(s['exact']['numeric'], .5);self.assertEqual(float(s['numeric_cell_max'][0, 0]), np.inf)
        self.assertEqual(P.plain(dict(x=np.float32(np.inf), y=[np.int64(2), np.bool_(True)])), dict(x='inf', y=[2, True]))

    def test_tokens_matching_up_to_order(self):
        env, view = toy(3)
        for i in range(3):env['numeric'][i, 1:7, 0] = np.arange(6) / 10;env['categorical'][i, 1:7, 0] = np.arange(6)
        view['numeric'][:] = env['numeric'];view['categorical'][:] = env['categorical']
        view['numeric'][1, 1:7] = view['numeric'][1, 1:7][::-1].copy();view['categorical'][1, 1:7] = view['categorical'][1, 1:7][::-1].copy()   # permuted
        view['numeric'][2, 1:7] = view['numeric'][2, 1:7][::-1].copy()   # numeric rows permuted without their categorical rows: not a permutation
        s = P.diff_stats(env, view, tolerance=1e-5)
        self.assertAlmostEqual(s['order']['self'], 2 / 3);self.assertEqual(s['order']['opponent'], 1.);self.assertAlmostEqual(s['exact']['numeric'], 1 / 3)

    def test_the_redactions_apply_to_both_sides(self):
        env, view = toy(4)
        env['numeric'][0, 1, 39] = .3;env['numeric'][1, 8, 38] = .2   # exact private losses the view redacts
        env['numeric'][2, 9, 0] = 263 / 300;view['numeric'][2, 9, 0] = .88   # an exact fraction against the public percent
        env['numeric'][3, 7, 0] = .995;view['numeric'][3, 7, 0] = .99   # below max: capped at 99
        view['numeric'][3, 10, 0] = .07;env['numeric'][3, 10, 0] = np.float32(.07)   # float32 k/100 stays k
        view['numeric'][1, 2, 39] = .1   # LAST_DAMAGE_TAKEN off self0 is NOT redacted
        a, b = P.redact(env), P.redact(view)
        self.assertEqual((float(a['numeric'][0, 1, 39]), float(a['numeric'][1, 8, 38]), float(a['numeric'][2, 9, 0]), float(a['numeric'][3, 7, 0])),
            (0., 0., float(np.float32(.88)), float(np.float32(.99))))
        np.testing.assert_array_equal(P.public_percent([0, 1e-9, .5, .995, 1., np.nan]), [0, .01, .5, .99, 1., np.nan])
        s = P.diff_stats(a, b, tolerance=1e-5)
        self.assertEqual(s['exact']['numeric'], .75);self.assertEqual(int(np.argmax(s['numeric_cell_share'].reshape(-1))), 2 * 132 + 39)   # the self1 cell
        raw = P.diff_stats(env, view, tolerance=1e-5);self.assertEqual(raw['exact']['numeric'], 0.)
        self.assertEqual(P.redact(env, ())['numeric'].tolist(), env['numeric'].tolist())
        b2 = P.redact(env, ('self_hazard_credit',));self.assertEqual(float(b2['numeric'][0, 0, 115]), 0.)

    def test_the_report_redacts_both_sides_with_the_five_documented_redactions_by_default(self):
        self.assertEqual(set(P.DEFAULT_REDACTIONS), {'last_damage_taken', 'last_damage_dealt', 'hp_percent', 'self_hazard_credit', 'opp_hazard_credit'})
        env, view = toy(4);env, view = env['numeric'], view['numeric']
        env[0, 1, 39] = .3   # the env's side of a redacted cell (the opponent's exact own loss)
        view[1, 9, 38] = .2   # the view's side of a redacted cell
        env[2, 0, 115] = .05   # the opponent's exact Spikes losses: the fourth redaction
        env[3, 8, 0] = 263 / 300;view[3, 8, 0] = .88   # an exact fraction against the public percent
        env[3, 0, 116] = .02;view[3, 0, 116] = .0213   # the opponent's view of our Spikes losses: exact sum against per-hit percents
        with tempfile.TemporaryDirectory() as tmp:
            capture = synthetic_capture(Path(tmp) / 'capture', env, view);report = P.build_report([capture])
            with patch('sys.stdout', new_callable=io.StringIO):
                self.assertEqual(P.main(['report', '--capture', str(capture), '--out', str(Path(tmp) / 'one'), '--redact', 'hp_percent']), 0)
            one = P.read_json(Path(tmp) / 'one' / 'summary.json');markdown = (Path(tmp) / 'one' / 'summary.md').read_text()
        self.assertEqual(report['parity']['true']['exact']['numeric'], 0.)   # every position differs somewhere
        self.assertEqual(report['parity_redacted']['true']['exact']['observation_and_legal'], 1.)   # exact except the four, both sides redacted
        self.assertEqual(sorted(report['redactions']), sorted(P.DEFAULT_REDACTIONS))
        self.assertEqual((sorted(one['redactions']), one['parity_redacted']['true']['exact']['numeric']), (['hp_percent'], 0.))   # --redact selects
        self.assertIn('Exact match except the documented redactions', markdown)

    def test_field_116_is_compared_against_the_env_recomputation_else_within_the_rounding_tolerance(self):
        env, view = toy(4);env, view = env['numeric'], view['numeric']
        for i, (e, v) in enumerate(((.02, .0213), (.02, .0213), (.02, .03), (.0, .0))):env[i, 0, 116], view[i, 0, 116] = e, v
        with tempfile.TemporaryDirectory() as tmp:
            recomputed = synthetic_capture(Path(tmp) / 'r', env, view, recomputed=[.0213, .0212, np.nan, np.nan])
            earlier = synthetic_capture(Path(tmp) / 'e', env, view)
            exact = lambda d: [bool(x) for x in P.diff_stats(*P.redact_pair(P.side_of(P.load_captures([d])[3], 'env', mask=np.ones(4, bool)),
                P.side_of(P.load_captures([d])[3], 'view', 0, np.ones(4, bool)), P.DEFAULT_REDACTIONS, P.load_captures([d])[3]['env_opp_hazard_credit']),
                tolerance=1e-5)['observation_exact']]
            # recomputed: exact equality (a recomputation 1e-4 off differs although within the tolerance); else the tolerance (0.01 apart differs)
            self.assertEqual(exact(recomputed), [True, False, False, True])
            self.assertEqual(exact(earlier), [True, True, False, True])
            self.assertTrue(np.isnan(P.load_captures([earlier])[3]['env_opp_hazard_credit']).all())   # an earlier capture: no recomputation

    def test_the_recomputation_is_the_views_fill(self):
        from gpu_engine import constructor_worlds as cw, engine_device_public_port as port, engine_device_trackers as tr
        lines = ['|switch|p1a: Alpha|Alpha, L80|300/300', '|switch|p2a: Beta|Beta, L80|100/100', '|turn|1', '|switch|p1a: Gamma|Gamma, L80|299/300',
            '|-damage|p1a: Gamma|262/300|[from] Spikes', '|switch|p1a: Alpha|Alpha, L80|300/300', '|-damage|p1a: Alpha|263/300|[from] Spikes',
            '|switch|p1a: Eta|Eta, L80|10/300', '|-damage|p1a: Eta|0 fnt|[from] Spikes']
        z = jnp.zeros((1, 2, 6), jnp.int32);base = tr.Trackers(*([z] * 5), z, z, z, z, z, jnp.zeros((1, 2, 6, 7), jnp.int32), z > 0, z > 0, jnp.ones(1, jnp.int32),
            jnp.zeros(1, jnp.int32), jnp.zeros((1, 2), jnp.int32), jnp.zeros(1, jnp.int32), public_port=port.initialize(1))
        filled = cw.fill_owner_spikes(base, 0, cw.line_facts(lines, 0));pair = np.asarray(filled.public_port.hazards)[0, 0, 0]
        view = np.float32(np.clip(np.float32(pair[0] + pair[1]) / np.float32(6), 0, 1))   # what the encoder writes into field 116
        self.assertGreater(view, 0);self.assertEqual(P.recompute_opp_hazard_credit(lines, 0), view)
        self.assertEqual(P.recompute_opp_hazard_credit(lines[:3], 0), 0.)   # no hit
        self.assertTrue(np.isnan(P.recompute_opp_hazard_credit(['|-damage|p1a: Gamma|262/300|[from] Spikes'], 0)))   # no previous HP
        self.assertEqual((P.showdown_percent(299, 300), P.showdown_percent(300, 300), P.showdown_percent(1, 300), P.showdown_percent(0, 300)), (99, 100, 1, 0))

    def test_empty_kind(self):
        env, view = toy(0)
        s = P.diff_stats(env, view, tolerance=1e-5)
        self.assertEqual((s['positions'], s['exact']['numeric'], s['order']['self']), (0, None, None))
        self.assertEqual(P.ranked_cells(s, 'numeric', P.token_labels(), P.numeric_names()), [])

    def test_names_cover_the_v4_layout(self):
        labels, numeric, categorical = P.token_labels(), P.numeric_names(), P.categorical_names()
        self.assertEqual((len(labels), len(numeric), len(categorical)), (23, 132, 41))
        self.assertEqual((labels[0], labels[1], labels[7], labels[13], labels[17], labels[22]), ('field', 'self0', 'opp0', 'move0', 'switch0', 'tendency_stats'))
        self.assertEqual(len(set(numeric)), 132);self.assertFalse([n for n in numeric if n.startswith('legacy')])
        self.assertEqual((numeric[0], categorical[0], categorical[39], categorical[40]), ('HP_FRACTION', 'PRIMARY', 'LAST_USED_MOVE', 'TRACED_ABILITY'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'v.json';path.write_text(json.dumps(dict(tokens=['species:snorlax', 'move:surf'])))
            decode = P.vocabulary_decoder(path)
        self.assertEqual([decode(i) for i in (0, 1, 2, 3)], ['<pad>', 'species:snorlax', 'move:surf', '<oov:0>']);self.assertIsNone(P.vocabulary_decoder(None))


if __name__ == '__main__':
    unittest.main()
