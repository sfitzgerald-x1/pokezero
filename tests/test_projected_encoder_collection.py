"""Whole authored collections; fake adapters and references, no runtime launch."""
from contextlib import ExitStack
import gzip
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_projected_encoder_parity import bootstrap_dependencies, example, fp32
bootstrap_dependencies()
from pokezero.mirror_value_probe import canonical, digest
from pokezero.observation import PokeZeroObservationV0, ObservationPerspective
from pokezero.projected_encoder_collection import compare_projected_collection, _BattleAudit, _AuditedAdapter
from pokezero.recorded_encoder_corpus import _Prefix
from pokezero.recorded_encoder_parity import SHAPES
from pokezero.saved_player_projection import PROJECTION_SCOPE, STRUGGLE_WARNING_POLICY, PINNED_SHOWDOWN_REVISION
import test_recorded_encoder_inventory as fixtures


class FakeProjector:
    def __init__(self, *, battle_id):
        self.battle_id, self.prefix = battle_id, _Prefix()

    def project_nonrequest_delta(self, lines):
        start = self.prefix.count
        self.prefix = self.prefix.append(lines)
        owners = {seat: SimpleNamespace(seat=seat, lines=tuple(lines)+(f'|-message|authored {seat}',))
                  for seat in ('p1', 'p2')}
        return SimpleNamespace(battle_id=self.battle_id, for_seat=lambda seat: owners[seat],
            source_nonrequest_start=start, source_nonrequest_end=self.prefix.count,
            source_nonrequest_prefix_sha256=self.prefix.sha256(), counts={'source_lines': len(lines)},
            scope=PROJECTION_SCOPE, struggle_warning_policy=STRUGGLE_WARNING_POLICY,
            showdown_revision=PINNED_SHOWDOWN_REVISION, captured_player_wire=False,
            owning_stream_bytes_reconstructed=False, public_wire_qualified=False, fullencoder=False)


class FakeEnvironment:
    def __init__(self, battle, seat):
        self.battle, self.seat, self.closed, self.requests, self.lines = battle, seat, False, [], []

    def close(self):
        self.closed = True


class FakeAdapter:
    def __init__(self, seat):
        self.seat, self.resets, self.feeds, self.forwards = seat, [], [], []

    def reset(self, **kwargs):
        if kwargs in self.resets: raise AssertionError('fresh registration required')
        self.resets.append(kwargs)

    def feed(self, lines):
        if any(line.startswith('|request|') for line in lines): raise AssertionError('request leaked')
        self.feeds.append(lines)

    def encode(self, request, *, owner_side):
        if request['side']['id'] != self.seat or owner_side != int(self.seat[1])-1:
            raise AssertionError('foreign owning request')
        return example(seat=self.seat).actual

    def forward(self, observation):
        if set(observation) != set(SHAPES): raise AssertionError('metadata reached model')
        if any(value.flags.writeable for value in observation.values()): raise AssertionError('mutable model input')
        self.forwards.append(observation['numeric_features'][0, 0])
        return fp32()


def fake_feed(environments, batch, *, own_request_deltas, requested_seats, request_tail_verified):
    if request_tail_verified is not True: raise AssertionError('chronology not attested')
    out = {}
    for seat, environment in environments.items():
        environment.lines.append(batch.for_seat(seat).lines)
        environment.requests.extend(own_request_deltas[seat])
        if seat in requested_seats:
            out[seat] = PokeZeroObservationV0(**example(seat=seat).actual,
                schema_version='pokezero.observation.v4', perspective=ObservationPerspective.from_showdown_slot(seat, seat),
                metadata=dict(battle_id=environment.battle, player_id=seat, showdown_slot=seat,
                    opponent_showdown_slot='p2' if seat == 'p1' else 'p1', turn_number=3))
    return out


class ProjectedCollectionTests(unittest.TestCase):
    def prepare(self, root):
        fixture = fixtures.RecordedEncoderInventoryTests()
        original_rows = fixture.rows
        def authored_rows(battle):
            first, end = original_rows(battle)
            second = json.loads(canonical(first))
            second['decision_round_index'] = 1
            second['requests']['p2']['wait'] = True
            second['requested_players'], second['paired'] = ['p1'], False
            second['observations'].pop('p2'); second['actions'].pop('p2')
            delta = ['|', '|request|' + json.dumps(second['requests']['p2'])]
            second.update(protocol_start=first['protocol_end'], protocol_end=first['protocol_end']+len(delta),
                protocol_delta=delta, public_prefix_sha256=digest(first['protocol_delta']+delta))
            second['root_id'] = digest([battle['battle_id'], 1, second['public_prefix_sha256'], second['requests']])
            end.update(protocol_start=second['protocol_end'], protocol_end=second['protocol_end']+1)
            return [first, second, end]
        fixture.rows = authored_rows
        kwargs = fixture.collection(root)
        self.environments = []
        self.adapters = {seat: FakeAdapter(seat) for seat in ('p1', 'p2')}
        return kwargs

    def run_collection(self, root, kwargs, *, with_forward=False, feed=fake_feed, projector=FakeProjector):
        def factory(config, *, battle_id):
            result = {seat: FakeEnvironment(battle_id, seat) for seat in ('p1', 'p2')}
            self.environments.append(result)
            return result
        module = 'pokezero.projected_encoder_collection.'
        with ExitStack() as stack:
            for target in ('subprocess.run', 'subprocess.Popen', 'numpy.load'):
                stack.enter_context(patch(target, side_effect=AssertionError('runtime/NPZ forbidden')))
            stack.enter_context(patch(module+'new_player_reference_environments', side_effect=factory))
            stack.enter_context(patch(module+'feed_player_reference_boundary', side_effect=feed))
            stack.enter_context(patch(module+'SavedSourcePlayerProjector', projector))
            return compare_projected_collection(root, **kwargs, adapters=self.adapters,
                reference_config=object(), with_forward=with_forward)

    def test_whole_roster_two_reused_adapters_wait_asymmetry_and_new_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); kwargs = self.prepare(root)
            result = self.run_collection(root, kwargs, with_forward=True)
            self.assertEqual(result['metrics']['views'], 24)
            self.assertEqual(result['metrics']['model_forward_views'], 24)
            self.assertEqual(len(result['battles']), 8)
            self.assertTrue(result['metrics']['model_forward_complete'])
            self.assertFalse(result['metrics']['projected_comparison_numeric_criteria_met'])
            self.assertEqual([row['stream_id'] for row in self.adapters['p1'].resets], list(range(2, 18, 2)))
            self.assertEqual([row['stream_id'] for row in self.adapters['p2'].resets], list(range(3, 19, 2)))
            self.assertEqual([len(adapter.feeds) for adapter in self.adapters.values()], [16, 16])
            self.assertEqual([len(adapter.forwards) for adapter in self.adapters.values()], [32, 16])
            self.assertTrue(all(value == .25 for adapter in self.adapters.values() for value in adapter.forwards))
            self.assertEqual(result['metrics']['numeric_columns'][0]['mismatched_cells'], 0)
            for environments, report in zip(self.environments, result['battles']):
                self.assertTrue(all(env.closed for env in environments.values()))
                self.assertEqual(len(environments['p2'].requests), 2)
                self.assertTrue(json.loads(environments['p2'].requests[-1][len('|request|'):])['wait'])
                audit = report['projection_audit']
                self.assertEqual(audit['source_nonrequest_lines'], 3)
                self.assertEqual(audit['projection_counts'], {'source_lines': 3})
                self.assertEqual(audit['owners']['p1']['projected_lines'], 5)
                self.assertEqual(audit['owners']['p2']['own_request_lines'], 2)
                self.assertTrue(audit['audit_uses_actual_reference_and_adapter_snapshot'])
                for seat in ('p1', 'p2'):
                    prefix = _Prefix()
                    for lines in environments[seat].lines: prefix = prefix.append(lines)
                    self.assertEqual(audit['owners'][seat]['projected_prefix_sha256'], prefix.sha256())
            self.assertEqual(result['new_games'], 0)
            for key in ('captured_player_wire', 'owning_stream_bytes_reconstructed', 'public_wire_qualified',
                        'fullencoder', 'full_encoder_admitted', 'original_reference_values_compared'):
                self.assertIs(result[key], False)
            # No tensors, requests, or protocol records are returned in compact reports.
            blob = canonical(result)
            self.assertNotIn(b'categorical_ids', blob)
            self.assertNotIn(b'authored p1', blob)

    def test_no_forward_never_implies_complete_model_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); kwargs = self.prepare(root)
            result = self.run_collection(root, kwargs)
            self.assertFalse(result['metrics']['model_forward_complete'])
            self.assertFalse(any(adapter.forwards for adapter in self.adapters.values()))

    def test_metadata_mutation_and_shared_adapter_abort_whole_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); kwargs = self.prepare(root)
            self.adapters['p2'] = self.adapters['p1']
            with self.assertRaisesRegex(ValueError, 'independent adapters'):
                self.run_collection(root, kwargs)
            self.assertEqual(self.environments, [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); kwargs = self.prepare(root)
            calls = 0
            def mutate(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 16:
                    path = root/'cohort.json'
                    path.write_bytes(path.read_bytes()+b' ')
                return fake_feed(*args, **kwargs)
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                self.run_collection(root, kwargs, feed=mutate)
            self.assertTrue(all(env.closed for pair in self.environments for env in pair.values()))

    def test_authored_metadata_uses_actual_projector_and_two_production_reference_histories(self):
        # Production Python parsing/normalization/annotations/encoding, backed
        # exclusively by authored metadata. This is not champion-vocab parity.
        from pokezero.belief import CandidateSetSummary
        from pokezero.category_vocab import build_category_vocabulary
        from pokezero.dex import MoveInfo, ShowdownDex, SpeciesInfo
        from pokezero.local_showdown import LocalShowdownConfig
        from pokezero.observation import ObservationFeatureMasks
        from pokezero.saved_player_projection import RULES, new_player_reference_environments
        from pokezero.showdown import V4_REPLAY_OBSERVATION_SPEC
        species = {name.lower(): SpeciesInfo(id=name.lower(), name=name, types=('Normal',),
                   base_stats={'hp': 100, 'atk': 80, 'def': 90, 'spa': 100, 'spd': 100, 'spe': 80})
                   for name in ('Blissey', 'Gengar')}
        move = MoveInfo(id='thunderbolt', name='Thunderbolt', type='Electric', category='Special',
            gen3_category='Special', base_power=95, accuracy=100., priority=0, recoil=False, drain=False,
            heal=False, status=None, boosts={}, target='normal', selfdestruct=False, pp=16)
        dex = ShowdownDex(moves={'thunderbolt': move}, species=species, type_chart={})
        class AuthoredSource:
            def summarize(self, *, species, **kwargs):
                return CandidateSetSummary(species=species, candidate_count=1, uncertainty=.25,
                    possible_abilities=('Natural Cure',), possible_items=('Leftovers',),
                    possible_moves=('thunderbolt',), candidate_variants=({'variant_id': species+'-authored',
                        'moves': ('thunderbolt',), 'ability': 'Natural Cure', 'item': 'Leftovers', 'level': 80},))
        vocab = build_category_vocabulary(['request_kind:move', 'species:Blissey', 'species:Gengar',
            'move:thunderbolt', 'belief:possible_ability:naturalcure', 'belief:possible_item:leftovers'])
        config = LocalShowdownConfig(showdown_root='/authored-no-runtime', set_belief_source=True,
            category_vocab=vocab, observation_spec=V4_REPLAY_OBSERVATION_SPEC,
            feature_masks=ObservationFeatureMasks(exact_state=True, transition_token_budget=0,
                opponent_tendency_stats_block=True, tier2_residuals=True, tier2_investment=True,
                investment_belief_narrowing=False, item_belief_narrowing=False, feature_pack_last_move=True))
        fixture = fixtures.RecordedEncoderInventoryTests()
        original_rows = fixture.rows
        def rows(battle):
            first, end = original_rows(battle)
            for seat, name, health in (('p1', 'Blissey', '300/400'), ('p2', 'Gengar', '200/301')):
                first['requests'][seat] = {'side': {'id': seat, 'pokemon': [dict(ident=f'{seat}: {name}',
                    details=name+', L80', condition=health, active=True,
                    stats={'atk': 120, 'def': 130, 'spa': 180, 'spd': 160, 'spe': 140},
                    moves=['thunderbolt'], baseAbility='Natural Cure', item='Leftovers')]},
                    'active': [{'moves': [dict(id='thunderbolt', move='Thunderbolt', pp=16, maxpp=16,
                        target='normal', disabled=False)]}]}
            lines = ['|gen|3', '|gametype|singles', '|tier|[Gen 3] Random Battle',
                '|player|p1|Authored one|', '|player|p2|Authored two|', '|teamsize|p1|6', '|teamsize|p2|6',
                *['|rule|'+rule for rule in sorted(RULES)], '|start',
                '|switch|p1a: Blissey|Blissey, L80|300/400', '|switch|p2a: Gengar|Gengar, L80|200/301', '|turn|3',
                *['|request|'+json.dumps(first['requests'][seat]) for seat in ('p1', 'p2')]]
            first.update(protocol_delta=lines, protocol_end=len(lines), public_prefix_sha256=digest(lines))
            first['root_id'] = digest([battle['battle_id'], 0, digest(lines), first['requests']])
            end.update(protocol_start=len(lines), protocol_end=len(lines)+1)
            return [first, end]
        fixture.rows = rows
        environments, closed = [], []
        def factory(config, *, battle_id):
            result = new_player_reference_environments(config, battle_id=battle_id)
            for seat, env in result.items():
                original = env.close
                def close(original=original, env=env):
                    closed.append(env)
                    return original()
                env.close = close
            environments.append(result)
            return result
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory); kwargs = fixture.collection(root)
            for target in ('subprocess.run', 'subprocess.Popen', 'numpy.load'):
                stack.enter_context(patch(target, side_effect=AssertionError('runtime/NPZ forbidden')))
            stack.enter_context(patch('pokezero.category_vocab.logger.warning'))
            stack.enter_context(patch('pokezero.local_showdown.load_gen3_randbat_source_cached', return_value=AuthoredSource()))
            stack.enter_context(patch('pokezero.local_showdown.load_showdown_dex_cached', return_value=dex))
            stack.enter_context(patch('pokezero.projected_encoder_collection.new_player_reference_environments', side_effect=factory))
            result = compare_projected_collection(root, **kwargs,
                adapters={seat: FakeAdapter(seat) for seat in ('p1', 'p2')}, reference_config=config, with_forward=False)
            self.assertEqual(result['metrics']['views'], 16)
            self.assertTrue(any(row['mismatched_cells'] for row in result['metrics']['numeric_columns']))
            self.assertEqual(len(closed), 16)
            self.assertEqual(len({id(env) for env in closed}), 16)
            for pair in environments:
                for seat, env in pair.items():
                    self.assertEqual(set(env._parser.requests), {seat})
                    self.assertEqual(env._parser.hp_visibility[seat], 'exact')
                    self.assertEqual(env._parser.hp_visibility['p2' if seat == 'p1' else 'p1'], 'percentage')
                self.assertIsNot(pair['p1']._tier2_trackers['p1'], pair['p2']._tier2_trackers['p2'])
                self.assertIsNot(pair['p1']._investment_trackers['p1'], pair['p2']._investment_trackers['p2'])

    def test_late_reference_forward_and_cleanup_errors_abort_without_report(self):
        for mode in ('reference', 'forward', 'close'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); kwargs = self.prepare(root)
                calls = 0
                def feed(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 15:
                        if mode == 'reference': raise RuntimeError('late reference failure')
                        if mode == 'close':
                            env = args[0]['p1']
                            def close():
                                env.closed = True
                                raise RuntimeError('close failure')
                            env.close = close
                    return fake_feed(*args, **kwargs)
                if mode == 'forward':
                    original = self.adapters['p1'].forward
                    def forward(value):
                        if calls == 15: raise RuntimeError('late forward failure')
                        return original(value)
                    self.adapters['p1'].forward = forward
                with self.assertRaises((RuntimeError, BaseExceptionGroup)):
                    self.run_collection(root, kwargs, with_forward=True, feed=feed)
                self.assertEqual(len(self.environments), 8)
                self.assertTrue(all(env.closed for pair in self.environments for env in pair.values()))

    def test_second_pass_hashes_exact_open_descriptor_even_after_path_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); kwargs = self.prepare(root)
            target = (root/'battle-0007/boundaries.jsonl.gz').resolve()
            original = target.read_bytes()
            rows = [json.loads(line) for line in gzip.decompress(original).splitlines()]
            rows[0]['protocol_delta'][1] = '|-crit|p1a: Absol'
            prefix = []
            for row in rows[:-1]:
                prefix.extend(row['protocol_delta'])
                row['public_prefix_sha256'] = digest(prefix)
                row['root_id'] = digest([row['battle_id'], row['decision_round_index'], digest(prefix), row['requests']])
            alternate, saved = target.with_name('alternate.gz'), target.with_name('authentic.gz')
            alternate.write_bytes(gzip.compress(b''.join(canonical(row)+b'\n' for row in rows)))
            real_open, opens = Path.open, 0
            def switched(path, *args, **kwargs):
                nonlocal opens
                if path == target and args and args[0] == 'rb':
                    opens += 1
                    if opens == 2:
                        os.replace(target, saved); os.replace(alternate, target)
                        stream = real_open(path, *args, **kwargs)
                        os.replace(target, alternate); os.replace(saved, target)
                        return stream
                return real_open(path, *args, **kwargs)
            with patch.object(Path, 'open', switched), self.assertRaisesRegex(ValueError, 'replayed projected.*SHA'):
                self.run_collection(root, kwargs)
            self.assertEqual(opens, 2)
            self.assertEqual(target.read_bytes(), original)
            self.assertTrue(all(env.closed for pair in self.environments for env in pair.values()))

    def test_scope_flags_and_dynamic_snapshot_handoff_are_not_merely_reported(self):
        for field, value in (('scope', 'other'), ('struggle_warning_policy', 'fabricated'),
                             ('fullencoder', True), ('public_wire_qualified', 0), ('captured_player_wire', True),
                             ('owning_stream_bytes_reconstructed', True), ('showdown_revision', 'other'),
                             ('counts', {'x': True})):
            class BadProjector(FakeProjector):
                def project_nonrequest_delta(self, lines):
                    result = super().project_nonrequest_delta(lines)
                    setattr(result, field, value)
                    return result
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); kwargs = self.prepare(root)
                with self.assertRaises(ValueError): self.run_collection(root, kwargs, projector=BadProjector)
                self.assertTrue(all(env.closed for pair in self.environments for env in pair.values()))
                self.assertFalse(any(adapter.feeds for adapter in self.adapters.values()))
        audit = _BattleAudit('authored')
        original = ('|-message|one',)
        audit.pending = {'p1': original}
        adapter = FakeAdapter('p1')
        wrapper = _AuditedAdapter(adapter, 'p1', audit)
        with self.assertRaisesRegex(ValueError, 'exact audited'):
            wrapper.feed(tuple(list(original)))
        self.assertFalse(adapter.feeds)
        wrapper.feed(original)
        self.assertEqual(adapter.feeds, [original])
