"""Ordered batches must agree with legacy steps, including every rejection."""
from dataclasses import fields
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import random
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, LocalShowdownError
from pokezero.showdown import _ReplayParser
from pokezero.showdown_fixture import FixturePokemon, pack_team
from pokezero.mcts_eval.paper_reference_staged_chance import _condition_chance_trials
from pokezero.mcts_eval.paper_reference import ReferenceRefusal, SamplingDeadlineExceeded


def public_history(replay):
    return tuple(event.raw_line for event in replay.public_events
        if event.raw_line not in ('', '|', "|message|The battle's RNG was reset."))


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(row) for key, row in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(row) for row in value
            if not (isinstance(row, str) and row.startswith('|t:|'))]
    return value


def annotation_state(value):
    if isinstance(value,dict):
        return {key:annotation_state(row) for key,row in value.items() if key!='_dex'}
    if isinstance(value,(tuple,list)):
        return tuple(annotation_state(row) for row in value)
    if isinstance(value,(set,frozenset)):
        return frozenset(annotation_state(row) for row in value)
    if hasattr(value,'__dict__'):
        return (type(value).__name__,annotation_state(vars(value)))
    return value


class RetryProjectionTests(unittest.TestCase):
    def test_bridge_projection_matches_parser_or_stops_without_rejection(self):
        module = (Path(__file__).resolve().parents[1] /
            'scripts/battle_bridge_conditioning_batch.mjs').as_uri()
        histories = [
            ["|message|The battle's RNG was reset.", '|', '|t:|123',
                '|move|p1a: Lugia|Psychic|p2a: Shuckle', '|-damage|p2a: Shuckle|97/198 slp',
                '|cant|p2a: Shuckle|slp', '|upkeep', '|turn|17'],
            ['|request|{"side":{"id":"p1"}}', '>reseed 1,2,3,4',
                '|-curestatus|p2a: Shuckle|slp', '|-start|p1a: Lugia|Encore', '|turn|17'],
            ['|faint|p2a: Shuckle', '|win|PokeZero p1'],
            ['|switch|p2a: Floof|Feraligatr, L78|100/100', '|-enditem|p2a: Floof|Salac Berry|[eat]'],
            ['|upkeep|extra', '|turn|17'], ['|upkeep ', '|turn|17'],
            ['|turn|16'], ['|turn|18'], ['|turn|017'], ['|turn|17|extra'],
            ['\u0085|cant|p2a: Shuckle|slp\u0085'], [''],
        ]
        result = subprocess.run(['node', '--input-type=module', '-e',
            'import {retryPublicSuffix} from ' + json.dumps(module) + ';\n' +
            'console.log(JSON.stringify(' + json.dumps(histories) + '.map(x=>retryPublicSuffix(x,16))));'],
            check=True, text=True, capture_output=True)
        projections = json.loads(result.stdout)
        for lines, projected in zip(histories, projections):
            if projected is None:
                continue  # Uncertain means return this candidate to Python, not reject it.
            parser = _ReplayParser()
            parser.turn_number = 16
            parser.feed(lines)
            self.assertEqual(tuple(projected), public_history(parser), lines)
        self.assertTrue(all(projections[index] is None for index in range(4, 11)))


class BatchRngOwnershipTests(unittest.TestCase):
    def test_first_match_rewinds_only_speculative_copy_not_original_stream(self):
        for position in (1, 3, 8, 9, 17):
            rng = random.Random(91)
            expected = random.Random(91)
            consumed_seeds = [expected.getrandbits(64) for _ in range(position)]
            calls, attempted = [], []
            def batch(snapshot, actions, *, chance_seeds, expected_history, deadline_at):
                calls.append(list(chance_seeds))
                count = min(position-len(attempted), len(chance_seeds))
                attempted.extend(chance_seeds[:count])
                return dict(consumed=count, matched=len(attempted)==position, deadline_reached=False)
            factory = SimpleNamespace(env=SimpleNamespace(conditioning_batch_from_search_snapshot=batch),
                conditioning_batch_size=8, sampling_deadline_at=time.perf_counter()+10,
                check_sampling_deadline=lambda:None)
            seed, count = _condition_chance_trials(factory, None, {}, ('target',), rng, subject='p1')
            self.assertEqual((seed, count), (consumed_seeds[-1], position))
            self.assertEqual(attempted, consumed_seeds)
            self.assertEqual(rng.getstate(), expected.getstate())
            self.assertEqual(len(calls), (position+7)//8)

    def test_deadline_partial_batch_advances_only_two_attempted_seeds(self):
        rng, expected = random.Random(17), random.Random(17)
        expected.getrandbits(64); expected.getrandbits(64)
        expired = [False]
        def check():
            if expired[0]:
                raise SamplingDeadlineExceeded('deadline')
        def batch(*args, **kwargs):
            expired[0] = True
            return dict(consumed=2, matched=False, deadline_reached=True)
        factory = SimpleNamespace(env=SimpleNamespace(conditioning_batch_from_search_snapshot=batch),
            conditioning_batch_size=8, sampling_deadline_at=time.perf_counter()+10, check_sampling_deadline=check)
        with self.assertRaises(SamplingDeadlineExceeded):
            _condition_chance_trials(factory, None, {}, (), rng, subject='p1')
        self.assertEqual(rng.getstate(), expected.getstate())

    def test_all_misses_keep_retry_cap_and_original_stream(self):
        rng, expected = random.Random(21), random.Random(21)
        attempted = []
        def batch(*args, chance_seeds, **kwargs):
            attempted.extend(chance_seeds)
            return dict(consumed=len(chance_seeds), matched=False, deadline_reached=False)
        factory = SimpleNamespace(env=SimpleNamespace(conditioning_batch_from_search_snapshot=batch),
            conditioning_batch_size=8, sampling_deadline_at=time.perf_counter()+10, check_sampling_deadline=lambda:None)
        self.assertIsNone(_condition_chance_trials(factory, None, {}, (), rng, subject='p1'))
        self.assertEqual(len(attempted), 2048)
        self.assertEqual(attempted, [expected.getrandbits(64) for _ in range(2048)])
        self.assertEqual(rng.getstate(), expected.getstate())

    def test_invalid_batch_size_or_absent_deadline_never_draws_original_rng(self):
        for size, deadline in ((True, 1), (0, 1), (17, 1), (8, None), (8, float('inf'))):
            rng = random.Random(91)
            state = rng.getstate()
            factory = SimpleNamespace(conditioning_batch_size=size, sampling_deadline_at=deadline,
                check_sampling_deadline=lambda:None)
            with self.assertRaises(ReferenceRefusal):
                _condition_chance_trials(factory, None, {}, (), rng, subject='p1')
            self.assertEqual(rng.getstate(), state)


@requires_showdown()
class ConditioningBatchIntegrationTests(unittest.TestCase):
    def test_auxiliary_trace_skips_python_projection_not_native_outcomes(self):
        from pokezero.mcts_eval.paper_reference_seed_bank import validate_native_trace
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=17,start_override=self.override())
            env.step({'p1':0,'p2':1})
            resident=env.snapshot_for_search()
            actions={'p1':0,'p2':1}
            history=public_history(resident.replay)
            for seed in (0,17,101,2**64-1):
                ordinary=env.conditioning_batch_from_search_snapshot(resident,actions,
                    chance_seeds=[seed],expected_history=history,
                    deadline_at=time.perf_counter()+10,fixed_pool=True,trace_chance=True)
                expected_native=normalized(env.snapshot().bridge_snapshot)
                expected_history=public_history(env.snapshot().replay)
                env.restore_search_snapshot(resident)
                before=env._local_snapshot(bridge_snapshot={'snapshot_id':'python-only'})
                with mock.patch.object(env,'_restore_local_snapshot_state',
                        side_effect=AssertionError('unused Python restore')), \
                     mock.patch.object(env,'_apply_event',
                        side_effect=AssertionError('unused Python event application')):
                    auxiliary=env.conditioning_batch_from_search_snapshot(resident,actions,
                        chance_seeds=[seed],expected_history=history,
                        deadline_at=time.perf_counter()+10,fixed_pool=True,
                        trace_chance=True,auxiliary_trace_only=True)
                after=env._local_snapshot(bridge_snapshot={'snapshot_id':'python-only'})
                self.assertEqual(before.replay,after.replay)
                self.assertEqual(before.protocol_lines,after.protocol_lines)
                self.assertEqual(before.latest_requests,after.latest_requests)
                self.assertEqual(before.request_history,after.request_history)
                self.assertEqual(before.belief_engine.snapshot(),after.belief_engine.snapshot())
                self.assertEqual(annotation_state(before.annotation_cache),
                                 annotation_state(after.annotation_cache))
                self.assertIsNone(auxiliary['step'])
                for key in ('consumed','matched','deadline_reached','matching_indices',
                        'first_public_mismatches','chance_traces'):
                    self.assertEqual(auxiliary[key],ordinary[key])
                validate_native_trace(auxiliary['chance_traces'][0],seed)
                # No generic snapshot/observation is read from the deliberately
                # unprojected shell. Actual transition restores both sides first.
                env.step_from_search_snapshot_for_conditioning(resident,actions,chance_seed=seed)
                self.assertEqual(normalized(env.snapshot().bridge_snapshot),expected_native)
                self.assertEqual(public_history(env.snapshot().replay),expected_history)
            env.release_search_snapshot(resident)

    def test_auxiliary_trace_flag_requires_one_owned_fixed_traced_trial(self):
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=17,start_override=self.override())
            resident=env.snapshot_for_search()
            with mock.patch.object(env,'_send_command',side_effect=AssertionError('must not send')):
                for flag,fixed,traced,seeds in (
                        (1,True,True,[1]),(True,False,True,[1]),
                        (True,True,False,[1]),(True,True,True,[1,2])):
                    with self.assertRaises(ValueError):
                        env.conditioning_batch_from_search_snapshot(resident,{'p1':0,'p2':1},
                            chance_seeds=seeds,expected_history=public_history(resident.replay),
                            deadline_at=time.perf_counter()+10,fixed_pool=fixed,
                            trace_chance=traced,auxiliary_trace_only=flag)
            env.release_search_snapshot(resident)

    def test_native_encore_trace_guides_original_roll_and_preserves_outcome(self):
        from pokezero.mcts_eval.paper_reference_seed_bank import encore_hints, matches_hints
        override=BattleStartOverride(observation_format_id='gen3randombattle',player_teams={
            'p1':pack_team((FixturePokemon('Aipom',('Growl','Tackle'),ability='Run Away'),)),
            'p2':pack_team((FixturePokemon('Wobbuffet',('Encore','Splash'),ability='Shadow Tag'),))})
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=223,start_override=override)
            resident=env.snapshot_for_search(); before=public_history(resident.replay)
            for seed in (0,17,101):
                env.step_from_search_snapshot_for_conditioning(resident,{'p1':0,'p2':0},chance_seed=seed)
                original=normalized(env.snapshot().bridge_snapshot)
                for _ in range(7):
                    if 'encore' not in env.snapshot().bridge_snapshot['battle']['sides'][0]['pokemon'][0]['volatiles']:break
                    env.step({'p1':0,'p2':1})
                expected=public_history(env.snapshot().replay)
                result=env.conditioning_batch_from_search_snapshot(resident,{'p1':0,'p2':0},
                    chance_seeds=[seed],expected_history=expected,deadline_at=time.perf_counter()+10,
                    fixed_pool=True,trace_chance=True)
                self.assertEqual(normalized(env.snapshot().bridge_snapshot),original)
                trace=result['chance_traces'][0]
                hints=encore_hints(trace,before,expected)
                self.assertEqual(len(hints),1, repr((trace,expected)))
                self.assertTrue(matches_hints(seed,hints))
                self.assertEqual(trace[hints[0]['ordinal']-1]['volatile']['ident'],'p1a: Aipom')
            env.release_search_snapshot(resident)

    def test_cross_critical_branch_uses_native_base_not_a_doubled_normal_base(self):
        from pokezero.mcts_eval.paper_reference_seed_bank import (chance_hints,
            critical_alignment_hints,matches_hints)
        override=BattleStartOverride(observation_format_id='gen3randombattle',player_teams={
            'p1':pack_team((FixturePokemon('Lugia',('Psychic',),ability='Pressure',level=74),)),
            'p2':pack_team((FixturePokemon('Shuckle',('Splash',),ability='Sturdy',level=100),))})
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=17,start_override=override)
            resident=env.snapshot_for_search();before=public_history(resident.replay)
            branches={}
            for seed in range(32):
                result=env.conditioning_batch_from_search_snapshot(resident,{'p1':0,'p2':0},
                    chance_seeds=[seed],expected_history=before,deadline_at=time.perf_counter()+10,
                    fixed_pool=True,trace_chance=True)
                trace=result['chance_traces'][0]
                roll=next(row for row in trace if row.get('damage') and 'base_damage' in row['damage'])
                branches.setdefault(roll['damage']['critical'],(seed,trace,public_history(env.snapshot().replay)))
                if len(branches)==2:break
            self.assertEqual(set(branches),{False,True})
            normal=branches[False][1];critical_seed,critical,expected=branches[True]
            old_branch_hints=chance_hints(normal,before,expected)
            self.assertFalse(any(h['kind']=='damage' for h in old_branch_hints))
            self.assertTrue(critical_alignment_hints(normal,before,expected))
            native_hints=chance_hints(critical,before,expected)
            self.assertTrue(any(h['kind']=='damage' for h in native_hints))
            self.assertTrue(matches_hints(critical_seed,native_hints))
            self.assertEqual(critical_alignment_hints(critical,before,expected),[])
            env.release_search_snapshot(resident)

    def test_native_trace_reports_actual_critical_damage_without_changing_outcomes(self):
        from pokezero.mcts_eval.paper_reference_seed_bank import chance_hints, matches_hints
        override = BattleStartOverride(observation_format_id='gen3randombattle', player_teams={
            'p1': pack_team((FixturePokemon('Tauros', ('Slash',), ability='Intimidate', level=50),)),
            'p2': pack_team((FixturePokemon('Snorlax', ('Splash',), ability='Immunity', level=100),))})
        criticals=0
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=17,start_override=override)
            resident=env.snapshot_for_search()
            for seed in range(24):
                result=env.conditioning_batch_from_search_snapshot(resident,{'p1':0,'p2':0},
                    chance_seeds=[seed],expected_history=public_history(resident.replay),
                    deadline_at=time.perf_counter()+10,fixed_pool=True,trace_chance=True)
                trace=result['chance_traces'][0]
                roll=next(row for row in trace if row.get('damage') and 'base_damage' in row['damage'])
                logged=any(line.startswith('|-crit|') for line in public_history(env.snapshot().replay))
                self.assertEqual(roll['damage']['critical'],logged)
                hints=chance_hints(trace,public_history(resident.replay),public_history(env.snapshot().replay))
                self.assertTrue(any(h['kind']=='damage' for h in hints), 'native argument format lost damage guidance')
                self.assertTrue(matches_hints(seed,hints), 'native executed seed disagrees with its own guidance')
                criticals+=logged
            self.assertGreater(criticals,0)
            env.release_search_snapshot(resident)

    def test_observational_chance_trace_preserves_hidden_state_public_state_and_rng(self):
        from pokezero.mcts_eval.paper_reference_seed_bank import validate_native_trace
        seeds = [0,1,17,101,2**64-1]
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=17,start_override=self.override())
            env.step({'p1':0,'p2':1})
            resident=env.snapshot_for_search();actions={'p1':0,'p2':1}
            for seed in seeds:
                env.step_from_search_snapshot_for_conditioning(resident,actions,chance_seed=seed)
                original=normalized(env.snapshot().bridge_snapshot)
                expected=public_history(env.snapshot().replay)
                result=env.conditioning_batch_from_search_snapshot(resident,actions,chance_seeds=[seed],
                    expected_history=expected,deadline_at=time.perf_counter()+10,fixed_pool=True,trace_chance=True)
                self.assertEqual(result['consumed'],1)
                trace=result['chance_traces'][0]
                self.assertTrue(trace);validate_native_trace(trace,seed)
                self.assertEqual(normalized(env.snapshot().bridge_snapshot),original)
                self.assertEqual(public_history(env.snapshot().replay),expected)
                # Subsequent original steps also see no installed tracing hooks.
                env.step_from_search_snapshot_for_conditioning(resident,actions,chance_seed=seed)
                self.assertEqual(normalized(env.snapshot().bridge_snapshot),original)
            env.release_search_snapshot(resident)

    def test_fixed_pool_processes_all_32_seeds_and_matches_legacy_prefixes(self):
        seeds = list(range(32))
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=17, start_override=self.override())
            env.step({'p1': 0, 'p2': 1})
            full, resident = env.snapshot(), env.snapshot_for_search()
            actions = {'p1': 0, 'p2': 1}
            histories, terminals = [], []
            for seed in seeds:
                env.restore(full); env.reseed_simulator_rng(seed); env.step(actions)
                histories.append(public_history(env.snapshot().replay))
                terminals.append(env.terminal() is not None)
            target = next(h for h, terminal in zip(histories, terminals) if not terminal)
            expected = [i for i, h in enumerate(histories) if h == target and not terminals[i]]
            before = env.root_puct_bridge_timing_snapshot()['bridge_round_trip_count']
            result = env.conditioning_batch_from_search_snapshot(resident, actions, chance_seeds=seeds,
                expected_history=target, deadline_at=time.perf_counter()+10, fixed_pool=True)
            self.assertEqual(result['consumed'], 32)
            self.assertEqual(result['matching_indices'], expected)
            self.assertEqual(env.root_puct_bridge_timing_snapshot()['bridge_round_trip_count']-before, 1)
            self.assertFalse(result['deadline_reached'])
            # A future public suffix is a prefix potential, not a demand that
            # the hypothetical transition jump past its current boundary.
            result = env.conditioning_batch_from_search_snapshot(resident, actions, chance_seeds=seeds,
                expected_history=target+('|message|future',), deadline_at=time.perf_counter()+10, fixed_pool=True)
            self.assertEqual(result['matching_indices'], expected)
            env.release_search_snapshot(resident)

    def config(self):
        return LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)

    def override(self):
        return BattleStartOverride(observation_format_id='gen3randombattle', player_teams={
            'p1': pack_team((FixturePokemon('Tauros', ('Body Slam', 'Earthquake'), ability='Intimidate', level=76),)),
            'p2': pack_team((FixturePokemon('Snorlax', ('Rest', 'Body Slam'), ability='Immunity', level=71),))})

    def contract(self, env):
        states = {}
        for player in ('p1', 'p2'):
            state = env.public_materialization_state(player)
            states[player] = {field.name: (state.belief_engine.snapshot() if field.name == 'belief_engine'
                else getattr(state, field.name)) for field in fields(state)}
        return (states, {player: env.observe(player) for player in env.requested_players()},
            normalized(env.snapshot().bridge_snapshot))

    def test_first_match_and_all_misses_preserve_complete_public_hidden_and_lazy_views(self):
        seeds = [0, 1, 17, 101, 2**64-1, 91, 213, 888]
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=17, start_override=self.override())
            env.step({'p1': 0, 'p2': 1})
            full, resident = env.snapshot(), env.snapshot_for_search()
            actions = {'p1': 0, 'p2': 1}
            histories, expected = [], []
            for seed in seeds:
                env.restore(full); env.reseed_simulator_rng(seed); env.step(actions)
                histories.append(public_history(env.public_materialization_state('p1').replay))
                expected.append(self.contract(env))
            self.assertGreater(len(set(histories)), 1)
            for target in (0, 3, 7):
                first = histories.index(histories[target])
                before = env.root_puct_bridge_timing_snapshot()['bridge_round_trip_count']
                result = env.conditioning_batch_from_search_snapshot(resident, actions,
                    chance_seeds=seeds, expected_history=histories[target], deadline_at=time.perf_counter()+10)
                self.assertEqual(result['consumed'], first+1)
                self.assertTrue(result['matched'])
                self.assertFalse(result['deadline_reached'])
                self.assertEqual(result['step'].observations, {})
                self.assertEqual(env.root_puct_bridge_timing_snapshot()['bridge_round_trip_count']-before, 1)
                self.assertEqual(self.contract(env), expected[first])
            result = env.conditioning_batch_from_search_snapshot(resident, actions,
                chance_seeds=seeds, expected_history=histories[0]+('|message|not observed',),
                deadline_at=time.perf_counter()+10)
            self.assertEqual(result['consumed'], len(seeds))
            self.assertFalse(result['matched'])
            self.assertEqual(self.contract(env), expected[-1])
            env.release_search_snapshot(resident)
            with self.assertRaisesRegex(LocalShowdownError, 'Unknown search snapshot'):
                env.conditioning_batch_from_search_snapshot(resident, actions,
                    chance_seeds=[1], expected_history=histories[0], deadline_at=time.perf_counter()+10)

    def test_terminal_rewards_are_not_accepted_as_conditioned_match(self):
        override = BattleStartOverride(player_teams={
            'p1': pack_team((FixturePokemon('Mewtwo', ('Psychic',), ability='Pressure', level=100),)),
            'p2': pack_team((FixturePokemon('Magikarp', ('Splash',), ability='Swift Swim', level=1),))})
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=17, start_override=override)
            full, resident = env.snapshot(), env.snapshot_for_search()
            env.restore(full); env.reseed_simulator_rng(999)
            expected = env.step({'p1': 0, 'p2': 0})
            target = public_history(env.snapshot().replay)
            result = env.conditioning_batch_from_search_snapshot(resident, {'p1': 0, 'p2': 0},
                chance_seeds=[998, 999], expected_history=target, deadline_at=time.perf_counter()+10)
            self.assertEqual(result['consumed'], 2)
            self.assertFalse(result['matched'])
            self.assertEqual(result['step'].terminal, expected.terminal)
            self.assertEqual(result['step'].rewards, expected.rewards)
            env.release_search_snapshot(resident)

    def test_invalid_arguments_expired_deadline_and_live_battle_never_send(self):
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=17, start_override=self.override())
            resident = env.snapshot_for_search()
            history = public_history(resident.replay)
            with mock.patch.object(env, '_send_command', side_effect=AssertionError('must not send')):
                for seeds in ([], [1]*17, [True], ['1']):
                    with self.assertRaises(ValueError):
                        env.conditioning_batch_from_search_snapshot(resident, {'p1': 0, 'p2': 0},
                            chance_seeds=seeds, expected_history=history, deadline_at=time.perf_counter()+10)
                for deadline in (float('nan'), float('inf'), True):
                    with self.assertRaises(ValueError):
                        env.conditioning_batch_from_search_snapshot(resident, {'p1': 0, 'p2': 0},
                            chance_seeds=[1], expected_history=history, deadline_at=deadline)
                self.assertEqual(env.conditioning_batch_from_search_snapshot(resident, {'p1': 0, 'p2': 0},
                    chance_seeds=[1], expected_history=history, deadline_at=time.perf_counter()-1)['consumed'], 0)
                env._search_snapshot_permitted = False
                with self.assertRaisesRegex(LocalShowdownError, 'sampled search world'):
                    env.conditioning_batch_from_search_snapshot(resident, {'p1': 0, 'p2': 0},
                        chance_seeds=[1], expected_history=history, deadline_at=time.perf_counter()+10)
            env._search_snapshot_permitted = True
            env.release_search_snapshot(resident)

    def test_corrupted_order_rejection_and_final_stream_receipts_fail_closed(self):
        with LocalShowdownEnv(self.config()) as env:
            env.reset_with_start_override(seed=17, start_override=self.override())
            resident = env.snapshot_for_search()
            target = public_history(resident.replay)+('|message|impossible target',)
            original = env._bridge_request_event
            captured = []
            def capture(*args, **kwargs):
                value = original(*args, **kwargs)
                captured.append(deepcopy(value))
                return value
            with mock.patch.object(env, '_bridge_request_event', side_effect=capture):
                env.conditioning_batch_from_search_snapshot(resident, {'p1':0,'p2':1},
                    chance_seeds=[1,17], expected_history=target, deadline_at=time.perf_counter()+10)
            valid = captured[-1]
            changes = [
                lambda value:value['trials'][0].update(index=1),
                lambda value:value['trials'][0].update(matched=True),
                lambda value:value['trials'][0].update(uncertain=True),
                lambda value:value.update(matched=True),
                lambda value:value['trials'][-1]['publicLines'].append('|message|different final stream'),
                lambda value:value['events'][-1].update(battleId='other-battle'),
                lambda value:value.update(trials=[],events=[],deadlineReached=False),
            ]
            for change in changes:
                broken = deepcopy(valid)
                change(broken)
                with mock.patch.object(env, '_bridge_request_event', return_value=broken):
                    with self.assertRaises(LocalShowdownError):
                        env.conditioning_batch_from_search_snapshot(resident, {'p1':0,'p2':1},
                            chance_seeds=[1,17], expected_history=target, deadline_at=time.perf_counter()+10)
            env.release_search_snapshot(resident)


if __name__ == '__main__':
    unittest.main()
