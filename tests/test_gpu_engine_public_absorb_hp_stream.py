"""Authored absorb-heal raw frames; no corpus, model or simulator access."""
from dataclasses import replace
import importlib.util
import json
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class AbsorbHPStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        from gpu_engine import public_belief, public_wire, public_inputs, public_tendency, v4_encoder
        from pokezero.randbat import (Gen3RandbatSource, Gen3RandbatSpeciesUniverse,
            Gen3RandbatVariant, RandbatSourceMetadata)
        from pokezero.dex import ShowdownDex, SpeciesInfo, MoveInfo
        from pokezero.category_vocab import CategoryVocabulary
        from test_gpu_engine_v4_encoder import V4EncoderTests
        V4EncoderTests.setUpClass()
        base = V4EncoderTests
        rows = {}
        profiles = (('Plusle', 'Plus', ('Thunderbolt', 'Surf')),
            ('Jolteon', 'Volt Absorb', ('Thunderbolt',)),
            ('Vaporeon', 'Water Absorb', ('Surf',)))
        for name, ability, moves in profiles:
            variant = Gen3RandbatVariant(name, 'synthetic', name, 'synthetic', 80,
                moves, ability, 'Leftovers')
            rows[name.lower()] = Gen3RandbatSpeciesUniverse(name, 80, (variant,))
        raw = Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle', 3,
            None, None, None, 'synthetic-absorb-hp-stream'), rows)
        source = public_belief.compile_source(raw)
        moves = dict(base.inputdex.moves)
        moves['thunderbolt'] = MoveInfo('thunderbolt', 'Thunderbolt', 'Electric', 'Special',
            'Special', 95, 100, 0, False, False, False, 'par', {}, 'normal', False, pp=15)
        species = dict(base.inputdex.species)
        for name, types in (('plusle', ('Electric',)), ('jolteon', ('Electric',)),
                             ('vaporeon', ('Water',))):
            species[name] = SpeciesInfo(name, name.title(), types,
                dict(hp=65, atk=65, **{'def': 65}, spa=110, spd=95, spe=130), 25.)
        dex = ShowdownDex(moves, species, base.inputdex.type_chart)
        labels = set(base.vocab.tokens) | {'move:thunderbolt', 'belief:possible_move:thunderbolt'}
        for name, ability, _ in profiles:
            labels |= {'species:' + name.lower(), 'ability:' + ability.lower().replace(' ', ''),
                'belief:possible_ability:' + ability.lower().replace(' ', '')}
        vocab = CategoryVocabulary(tuple(sorted(labels)))
        catalog = v4_encoder.compile_encoder(source, dex, vocab,
            checkpoint_sha256='a' * 64, feature_masks=base.masks)
        cls.jax, cls.j, cls.e, cls.w, cls.i, cls.t = jax, j, v4_encoder, public_wire, public_inputs, public_tendency
        cls.raw, cls.catalog, cls.vocab = raw, catalog, vocab
        cls.step = jax.jit(lambda state, chunk: v4_encoder.fold_chunk(state, chunk, catalog))
        cls.encode = jax.jit(lambda state, own, side: v4_encoder.encode(
            state, own, catalog, observer_sides=side))

    def initial(self, owner=0):
        return self.e.initialize(self.catalog, stream_ids=self.j.array([1], self.j.int32),
            owner_sides=np.array([owner], np.int32))

    def example(self, ability='Volt Absorb', *, owner=0, declared=False, of=True):
        name, move = ('Jolteon', 'Thunderbolt') if ability == 'Volt Absorb' else ('Vaporeon', 'Surf')
        condition = '79/100' if owner == 0 else '179/227'
        lines = ['|switch|p1a: Plusle|Plusle, L80|' + ('200/200' if owner == 0 else '100/100'),
            f'|switch|p2a: {name}|{name}, L80|' + ('55/100' if owner == 0 else '123/227'), '|turn|1']
        if declared:
            lines.append(f'|-ability|p2a: {name}|{ability}')
        lines.append(f'|move|p1a: Plusle|{move}|p2a: {name}')
        heal = f'|-heal|p2a: {name}|{condition}|[from] ability: {ability}'
        if of:
            heal += '|[of] p1a: Plusle'
        return lines, heal, name

    def projections(self, lines, host=None):
        return self.w.project_lines(host or self.w.WireState(), lines, **self.e.wire_kwargs(self.catalog))

    def chunks(self, projections, events=None):
        first = projections[0]
        bindings = {name: np.asarray(getattr(first, name), np.uint32) for name in (
            'species_binding', 'ability_binding', 'move_binding', 'item_binding', 'context_binding',
            'field_binding', 'active_binding', 'damage_binding', 'type_binding', 'sleep_binding',
            'toxic_binding', 'tendency_binding')}
        if events is None:
            events = [event for projection in projections for event in projection.events]
        return list(self.i.pack_chunks(np.array([1], np.int32), [events],
            source_binding=self.catalog.source.device.binding, **bindings))

    def before_heal(self, ability='Volt Absorb', *, owner=0, declared=False, of=True):
        lines, heal, name = self.example(ability, owner=owner, declared=declared, of=of)
        host, projections = self.projections(lines)
        state = self.initial(owner)
        for chunk in self.chunks(projections):
            state, error = type(self).step(state, chunk)
            self.assertEqual(error.tolist(), [0])
        _, frame = self.projections([heal], host)
        return state, frame, lines + [heal], name

    def same(self, first, second):
        for a, b in zip(self.jax.tree.leaves(first), self.jax.tree.leaves(second), strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def request(self, owner, name, ability):
        names = ('Plusle', name)
        moves = ('Thunderbolt', 'Surf') if owner == 0 else (('Thunderbolt',) if name == 'Jolteon' else ('Surf',))
        own_moves = []
        for move in moves:
            key = move.lower()
            maximum = int(self.catalog.dex.moves[key].pp * 8 // 5)
            own_moves.append(dict(id=key, pp=maximum - int(owner == 0), maxpp=maximum))
        return dict(rqid=2, side=dict(id=f'p{owner+1}', pokemon=[dict(
            ident=f'p{owner+1}: {names[owner]}', details=f'{names[owner]}, L80',
            condition='200/200' if owner == 0 else '179/227', active=True,
            ability='Plus' if owner == 0 else ability, item='Leftovers', moves=list(moves),
            stats=dict(atk=150, **{'def': 160}, spa=170, spd=180, spe=190))]),
            active=[dict(moves=own_moves)])

    def assert_source_parity(self, state, lines, owner, name, ability):
        from pokezero import showdown
        request = self.request(owner, name, ability)
        own = self.e.parse_own_requests(self.catalog, [request], owner_sides=[owner])
        encoded = type(self).encode(state, own, self.j.array([owner], self.j.int32))
        actual = encoded.observation
        self.assertEqual(actual.admitted.tolist(), [True])
        parser = showdown._ReplayParser('synthetic-absorb', complete_prefix=True,
            hp_visibility={f'p{side+1}': 'exact' if side == owner else 'percentage' for side in (0, 1)})
        for line in [*lines, '|request|' + json.dumps(request)]:
            parser._feed_line(line)
        replay = parser.snapshot()
        normalized = showdown.normalize_for_player(replay, player_id=f'p{owner+1}',
            configured_showdown_slot=f'p{owner+1}', format_id='gen3randombattle', set_source=self.raw)
        masks = replace(showdown.DEFAULT_OBSERVATION_FEATURE_MASKS, exact_state=True,
            opponent_tendency_stats_block=True, feature_pack_last_move=True, transition_token_budget=0)
        expected = showdown.observation_from_player_state(normalized, category_vocab=self.vocab,
            spec=showdown.V4_REPLAY_OBSERVATION_SPEC, dex=self.catalog.dex, feature_masks=masks)
        np.testing.assert_allclose(np.asarray(actual.numeric_features[0]), expected.numeric_features, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(np.asarray(actual.categorical_ids[0]), expected.categorical_ids)
        for field in ('token_type_ids', 'attention_mask', 'legal_action_mask'):
            np.testing.assert_array_equal(np.asarray(getattr(actual, field)[0]), getattr(expected, field))
        self.assertEqual(encoded.pending_numeric_columns.tolist(), [])
        self.assertFalse(bool(encoded.full_encoder_admitted[0]))

    def test_exact_percentage_both_absorbs_known_proc_and_no_of_match_source(self):
        for ability in ('Volt Absorb', 'Water Absorb'):
            for owner in (0, 1):
                for declared, of in ((False, False), (False, True), (True, False), (True, True)):
                    with self.subTest(ability=ability, owner=owner, declared=declared, of=of):
                        before, frame, lines, name = self.before_heal(ability, owner=owner, declared=declared, of=of)
                        events = frame[0].events
                        proc = [e for e in events if e.kind == self.i.Kind.ABILITY]
                        hp = [e for e in events if e.kind == self.i.Kind.CONDITION]
                        sleep = [e for e in events if e.kind == self.i.Kind.SLEEP]
                        closing = [e for e in events if e.kind == self.i.Kind.TENDENCY]
                        self.assertEqual([len(proc), len(hp), len(sleep), len(closing)], [1, 1, 1, 1])
                        aid = self.catalog.abilities.ability_id(ability)
                        self.assertEqual((proc[0].target_side, proc[0].target_slot, proc[0].payload[1]), (1, 0, aid))
                        self.assertEqual(closing[0].payload[0], self.t.OUTCOME)
                        self.assertEqual((closing[0].target_side, closing[0].payload[3], closing[0].payload[-1]), (1, 1, 0))
                        header = next(e for e in events if e.kind == self.i.Kind.RAW_BEGIN)
                        self.assertEqual(header.payload[4], 4 | (1 << 6) | (aid << 7))
                        self.assertEqual(header.payload[4] & 3, sleep[0].payload[3])
                        self.assertEqual(hp[0].payload[1:3], (79, 100) if owner == 0 else (179, 227))
                        after = before
                        for chunk in self.chunks(frame):
                            after, error = type(self).step(after, chunk)
                            self.assertEqual(error.tolist(), [0])
                        self.assertEqual(int(after.context.abilities.original[0, 1, 0]), aid)
                        self.assertEqual(int(after.mons.hp_numerator[0, 1, 0]), 79 if owner == 0 else 179)
                        self.assert_source_parity(after, lines, owner, name, ability)

    def test_semantic_absorb_id_mask_does_not_bless_other_abilities(self):
        mask = np.asarray(self.catalog.lifecycle.absorb_heal)
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertEqual(mask.shape, (len(self.catalog.abilities.names),))
        self.assertEqual({self.catalog.abilities.names[i] for i in np.flatnonzero(mask)},
            {'voltabsorb', 'waterabsorb'})

    def test_missing_duplicate_wrong_identity_and_downgraded_frames_roll_back(self):
        before, frame, _, _ = self.before_heal(declared=True)
        events = list(frame[0].events)
        proc = next(i for i, e in enumerate(events) if e.kind == self.i.Kind.ABILITY)
        hp = next(i for i, e in enumerate(events) if e.kind == self.i.Kind.CONDITION)
        sleep = next(i for i, e in enumerate(events) if e.kind == self.i.Kind.SLEEP)
        close = next(i for i, e in enumerate(events) if e.kind == self.i.Kind.TENDENCY)
        header = next(i for i, e in enumerate(events) if e.kind == self.i.Kind.RAW_BEGIN)
        def change(index, **fields):
            result = list(events); result[index] = replace(result[index], **fields); return result
        def header_flag(flag):
            payload = list(events[header].payload); payload[4] = flag
            return change(header, payload=tuple(payload))
        other = (self.t.OTHER, 0, 0, -1, 0, 0, 0, 9)
        downgraded = header_flag(events[header].payload[4] & 3)
        downgraded_other = list(downgraded)
        downgraded_other[close] = replace(events[close], target_side=-1, payload=other)
        wrong_id = list(events[proc].payload); wrong_id[1] = self.catalog.abilities.ability_id('Water Absorb')
        authoritative = list(events[proc].payload); authoritative[2] = 1
        wrong_hp_id = list(events[hp].payload); wrong_hp_id[0] += 1
        wrong_sleep_id = list(events[sleep].payload); wrong_sleep_id[0] += 1
        wrong_outcome = list(events[close].payload); wrong_outcome[-1] = 1
        wrong_close_target = list(events[close].payload); wrong_close_target[3] = 0
        cases = {
            'missing_proc_even_when_prior_declared': events[:proc] + events[proc+1:],
            'duplicate_proc': events[:proc] + [events[proc]] + events[proc:],
            'wrong_proc_target': change(proc, target_side=0),
            'wrong_proc_slot': change(proc, target_slot=1),
            'wrong_proc_id': change(proc, payload=tuple(wrong_id)),
            'authoritative_proc_mutation': change(proc, payload=tuple(authoritative)),
            'wrong_header_side': header_flag(4 | (self.catalog.abilities.ability_id('Volt Absorb') << 7)),
            'wrong_header_slot': header_flag(4 | (1 << 3) | (1 << 6) | (self.catalog.abilities.ability_id('Volt Absorb') << 7)),
            'wrong_header_id': header_flag(4 | (1 << 6) | (self.catalog.abilities.ability_id('Water Absorb') << 7)),
            'nonabsorb_header_id': header_flag(4 | (1 << 6) | (self.catalog.abilities.ability_id('Plus') << 7)),
            'missing_hp': events[:hp] + events[hp+1:],
            'missing_sleep': events[:sleep] + events[sleep+1:],
            'wrong_hp_target': change(hp, target_side=0),
            'wrong_hp_slot': change(hp, target_slot=1),
            'wrong_hp_id': change(hp, payload=tuple(wrong_hp_id)),
            'wrong_sleep_target': change(sleep, target_side=0),
            'wrong_sleep_slot': change(sleep, target_slot=1),
            'wrong_sleep_id': change(sleep, payload=tuple(wrong_sleep_id)),
            'downgraded_hp_header': downgraded,
            'downgraded_header_and_other_close': downgraded_other,
            'other_closing_kind': change(close, target_side=-1, payload=other),
            'wrong_closing_outcome': change(close, payload=tuple(wrong_outcome)),
            'wrong_closing_target': change(close, target_side=0, payload=tuple(wrong_close_target)),
        }
        for name, changed in cases.items():
            with self.subTest(case=name):
                changed = [replace(e, sequence=events[0].sequence + i) for i, e in enumerate(changed)]
                chunks = self.chunks(frame, changed)
                self.assertEqual(len(chunks), 1)
                after, error = type(self).step(before, chunks[0])
                self.assertNotEqual(error.tolist(), [0])
                self.same(before, after)

    def test_absorb_proof_survives_chunk_continuation_and_late_error_is_atomic(self):
        before, frame, _, _ = self.before_heal()
        events = list(frame[0].events)
        chunk = self.chunks(frame)[0]
        whole, error = type(self).step(before, chunk)
        self.assertEqual(error.tolist(), [0])
        for kind in (self.i.Kind.ABILITY, self.i.Kind.SLEEP):
            split = next(i for i, event in enumerate(events) if event.kind == kind) + 1
            def part(start, stop, more):
                words = np.zeros_like(np.asarray(chunk.words)); valid = np.zeros_like(np.asarray(chunk.valid))
                words[:, :stop-start] = np.asarray(chunk.words)[:, start:stop]
                valid[:, :stop-start] = True
                return chunk._replace(words=self.j.asarray(words), valid=self.j.asarray(valid),
                    more=self.j.array([more]))
            first, second = part(0, split, True), part(split, len(events), False)
            middle, error = type(self).step(before, first)
            self.assertEqual(error.tolist(), [0])
            self.assertTrue(bool(middle.cursor.needs_more[0]))
            self.assertEqual(int(middle.context.refresh.proc_seen[0]), 1)
            own = self.e.parse_own_requests(self.catalog,
                [self.request(0, 'Jolteon', 'Volt Absorb')], owner_sides=[0])
            incomplete = type(self).encode(middle, own, self.j.array([0], self.j.int32))
            self.assertEqual(incomplete.observation.admitted.tolist(), [False])
            after, error = type(self).step(middle, second)
            self.assertEqual(error.tolist(), [0])
            self.same(whole, after)
            close = next(i for i, event in enumerate(events) if event.kind == self.i.Kind.TENDENCY) - split
            bad = second._replace(words=second.words.at[0, close, 15].set(1))
            rejected, error = type(self).step(middle, bad)
            self.assertNotEqual(error.tolist(), [0])
            self.same(middle, rejected)


if __name__ == '__main__':
    unittest.main()
