"""Native integer/interval recurrence against an unaggregated independent oracle.

Run with the native extension built; absence of the extension is an error.
"""
from collections import defaultdict
from fractions import Fraction
import json
import random
import unittest
from unittest.mock import patch

import pokezero_search as native
from pokezero.joint_generator import JointGenerator, JointBudgetExceeded, ImpossibleJointEvidence, _context_after
from pokezero.joint_native import NativeJointPosterior, own_team_terminal_weights, draw_with_stage1_fallback, native_payload
from tests.test_depthn_joint_generator import fixture, brute


class NativeJointExactness(unittest.TestCase):
    def compare(self, parts, evidence, fixed, flag, terminal):
        exact = brute(*parts, evidence, fixed, flag, terminal)
        expected = sum(exact.values())
        kwargs = dict(evidence=evidence, fixed_slots=fixed, initial_wobbuffet=flag, terminal_weights=terminal)
        try:
            reference = JointGenerator(*parts, **kwargs)
            posterior = NativeJointPosterior(*parts, **kwargs)
        except ImpossibleJointEvidence:
            self.assertFalse(expected)
            return
        if not expected:
            with self.assertRaises(ImpossibleJointEvidence):
                reference.normalizer()
            with self.assertRaises(ImpossibleJointEvidence):
                posterior.solve(seconds=2)
            return
        result = posterior.solve(seconds=2)
        self.assertEqual(int(result['lower']), reference.normalizer())
        self.assertEqual(Fraction(int(result['lower']), reference.quantum ** reference.slots), expected)
        # All first-transition probabilities, not just a scalar normalizer.
        actual = defaultdict(Fraction)
        for gi, after, _, _, lower, upper in json.loads(posterior.native.branch_masses(0, 0, flag)):
            self.assertEqual(lower, upper)
            mass = Fraction(int(lower), int(result['lower']) * len(reference.groups[gi]))
            for bi in reference.groups[gi]:
                options = [(entry, weight) for entry, weight in reference._variant_options(bi, 0) if entry[2] == after]
                total = sum(w for _, w in options)
                for (species, vi, _), weight in options:
                    actual[species, vi] += mass * Fraction(weight, total)
        marginal = defaultdict(Fraction)
        for team, probability in exact.items():
            marginal[team[0]] += probability / expected
        self.assertEqual({k: v for k, v in actual.items() if v}, {k: v for k, v in marginal.items() if v})
        for species, weights in posterior.revealed_marginal_weights().items():
            expected_variants = defaultdict(Fraction)
            for team, probability in exact.items():
                for s, vi in team:
                    if s == species:
                        expected_variants[vi] += probability / expected
            self.assertEqual(tuple(Fraction(w, sum(weights)) for w in weights),
                             tuple(expected_variants[i] for i in range(len(weights))))
        # Exhaustive probabilities of complete ordered physical teams. Cache
        # prefix rows only in the test to avoid repeated FFI serialization.
        transitions = {}
        for team, probability in exact.items():
            if not probability:
                continue
            code, context, outgoing = 0, 0, flag
            selected, path_probability = set(), Fraction(1)
            for species, vi in team:
                key = code, context, outgoing
                if key not in transitions:
                    transitions[key] = json.loads(posterior.native.branch_masses(*key))
                rows = transitions[key]
                bi = next(i for i, b in enumerate(reference.bases) if species in b['forms'])
                gi = reference.base_group[bi]
                after = _context_after(context, parts[1]['species'][species][vi])
                row = next(r for r in rows if r[0] == gi and r[1] == after)
                path_probability *= Fraction(int(row[4]), sum(int(r[4]) for r in rows))
                remaining = sum(i not in selected for i in reference.groups[gi])
                options = [(entry, weight) for entry, weight in reference._variant_options(bi, context) if entry[2] == after]
                weight = next(w for (s, i, _), w in options if s == species and i == vi)
                path_probability *= Fraction(weight, remaining * sum(w for _, w in options))
                selected.add(bi); code, context, outgoing = row[2], row[1], row[3]
            self.assertEqual(path_probability, probability / expected)
        for team in posterior.draw_many(8, random.Random(73)):
            key = tuple((m.species, m.variant_index) for m in team)
            self.assertGreater(exact.get(key, 0), 0)

    def test_one_hundred_independent_tiny_oracles(self):
        rng = random.Random(20261116)
        for case in range(100):
            with self.subTest(case=case):
                parts = fixture(slots=rng.choice((2, 3)), rich=bool(case % 2))
                species = list(parts[1]['species'])
                picked = rng.sample(species, rng.randrange(4))
                evidence = {s: tuple(rng.randrange(5) for _ in parts[1]['species'][s]) for s in picked}
                fixed = {0: rng.choice(species)} if rng.randrange(3) == 0 else {}
                flag = bool(rng.randrange(2))
                terminal = rng.choice(((1, 1), (2, 7), (5, 0), (0, 3)))
                self.compare(parts, evidence, fixed, flag, terminal)

    def test_both_arm_terminal_likelihood(self):
        parts = fixture()
        team = (('a', 1), ('b', 1), ('c', 0))
        weights = own_team_terminal_weights(*parts, team)
        p = [brute(*parts, fixed={i: s for i, (s, _) in enumerate(team)},
                   evidence={s: tuple(int(j == vi) for j in range(len(parts[1]['species'][s]))) for s, vi in team},
                   initial_wobbuffet=flag) for flag in (False, True)]
        self.assertEqual(Fraction(*weights), sum(p[0].values()) / sum(p[1].values()))
        for position in (0, 2):
            own = list(team); own[position] = ('wobbuffet', 0)
            self.assertEqual(own_team_terminal_weights(*parts, own), (1, 0))
        self.compare(parts, {'a': (1, 1)}, {}, False, weights)

    def test_interval_encloses_exact_and_rejects_uncertified_draw(self):
        parts = fixture()
        exact = JointGenerator(*parts, evidence={'b': (1, 7)}, terminal_weights=(2, 9)).normalizer()
        posterior = NativeJointPosterior(*parts, evidence={'b': (1, 7)}, terminal_weights=(2, 9))
        with self.assertRaises(JointBudgetExceeded):
            posterior.solve(seconds=2, max_states=2, max_tv=Fraction(1, 100))
        self.assertLessEqual(int(posterior.result['lower']), exact)
        self.assertGreaterEqual(int(posterior.result['upper']), exact)
        with self.assertRaises(JointBudgetExceeded):
            posterior.draw_many(1, random.Random(1))
        with self.assertRaises(ValueError):
            posterior.native.draw_paths(1, 1)

    def test_reuse_legality_cache_with_new_evidence(self):
        parts = fixture(); cache = native.JointLegalityCache()
        for evidence in ({'b': (1, 1)}, {'b': (1, 0)}, {'a': (1, 0), 'b': (1, 0)}):
            p = NativeJointPosterior(*parts, evidence=evidence, legality_cache=cache)
            expected = JointGenerator(*parts, evidence=evidence).normalizer()
            self.assertEqual(int(p.solve(seconds=2)['lower']), expected)
        self.assertGreater(cache.stats()[1], 0)

    def test_certified_partial_distribution_and_frozen_frontier(self):
        parts = fixture(slots=2)
        kwargs = dict(evidence={'a': (1, 10**9)}, fixed_slots={0: 'a'})
        posterior = NativeJointPosterior(*parts, **kwargs)
        result = posterior.solve(seconds=2, max_states=2, max_tv=Fraction(1, 100))
        exact = JointGenerator(*parts, **kwargs).normalizer()
        self.assertFalse(result['exact'])
        self.assertLessEqual(int(result['lower']), exact)
        self.assertGreaterEqual(int(result['upper']), exact)
        self.assertLessEqual(posterior.tv_bound, Fraction(1, 100))
        before = posterior.native.branch_masses(0, 0, False)
        for team in posterior.draw_many(50, random.Random(17)):
            self.assertEqual((team[0].species, team[0].variant_index), ('a', 1))
        exact_distribution = brute(*parts, evidence=kwargs['evidence'], fixed=kwargs['fixed_slots'])
        removed = sum(p for team, p in exact_distribution.items() if team[0] == ('a', 0))
        actual_tv = removed / sum(exact_distribution.values())
        self.assertLessEqual(actual_tv, posterior.tv_bound)
        self.assertEqual(posterior.revealed_marginal_weights()['a'], (0, 1))
        payload = posterior.feature_payload()
        self.assertEqual(payload['rows']['a']['weights'], (0, 1))
        self.assertEqual(payload['rows']['a']['support'], (True, True))
        self.assertEqual(payload, posterior.feature_payload())
        self.assertEqual(posterior.native.branch_masses(0, 0, False), before)
        with self.assertRaises(ValueError):
            posterior.native.solve(2, 1000)
        with self.assertRaises(ValueError):
            posterior.native.branch_masses(256, 0, False)

    def test_stage1_fallback_preserves_rng(self):
        rng = random.Random(49)
        expected = random.Random(49)
        calls = []
        def stage1(count, draw_rng):
            calls.append(count)
            return tuple(draw_rng.randrange(10000) for _ in range(count))
        with patch('pokezero.joint_native.validate_pinned_catalog'):
            result = draw_with_stage1_fallback(*fixture(), count=32, rng=rng, stage1_sampler=stage1,
                battle_context=dict(owner='p1', own_initial_team=(('a', 0), ('b', 1), ('c', 0)), producer='shared-battle'),
                budget_seconds=.00001, draw_reserve_seconds=.000001)
        self.assertEqual(result.mode, 'stage1_fallback')
        self.assertEqual(result.teams, tuple(expected.randrange(10000) for _ in range(32)))
        self.assertEqual(calls, [32])
        self.assertIsNone(result.tv_bound)

    def test_late_feature_export_preserves_stage1_rng(self):
        context=dict(owner='p1',own_initial_team=(('a',0),('b',1),('c',0)),producer='shared-battle')
        rng=random.Random(75);expected=random.Random(75)
        with patch('pokezero.joint_native.validate_pinned_catalog'),patch.object(
                NativeJointPosterior,'feature_payload',side_effect=JointBudgetExceeded('late feature')):
            result=draw_with_stage1_fallback(*fixture(),count=2,rng=rng,export_features=True,
                stage1_sampler=lambda count,rng:tuple(rng.randrange(1000)for _ in range(count)),
                battle_context=context,budget_seconds=2,draw_reserve_seconds=.05)
        self.assertEqual(result.mode,'stage1_fallback')
        self.assertEqual(result.teams,tuple(expected.randrange(1000)for _ in range(2)))
        self.assertEqual(rng.getstate(),expected.getstate())

    def test_sampler_marginals_export_without_feature_prior(self):
        context=dict(owner='p1',own_initial_team=(('a',0),('b',1),('c',0)),producer='shared-battle')
        with patch('pokezero.joint_native.validate_pinned_catalog'):
            result=draw_with_stage1_fallback(*fixture(),count=2,rng=random.Random(6),export_posterior=True,
                stage1_sampler=lambda count,rng:(),battle_context=context,evidence={'a':(1,7)},
                budget_seconds=2,draw_reserve_seconds=.05)
        self.assertEqual(result.mode,'exact')
        self.assertTrue(result.features['rows']['a']['weights'])
        self.assertNotIn('prior_rows',result.features)

    def test_native_certificate_and_population_inputs_are_checked(self):
        payload = json.loads(native_payload(JointGenerator(*fixture())))
        payload['completable'] = True
        with self.assertRaisesRegex(ValueError, 'certificate catalog'):
            native.NativeJointGenerator(json.dumps(payload), native.JointLegalityCache())
        payload['completable'] = False
        payload['groups'][0]['size'] = 2**63
        with self.assertRaisesRegex(ValueError, 'population'):
            native.NativeJointGenerator(json.dumps(payload), native.JointLegalityCache())

    def test_native_boundary_rejects_positive_uncertified_mass(self):
        p = NativeJointPosterior(*fixture(slots=2), evidence={'a': (1, 1)}, fixed_slots={0: 'a'})
        with self.assertRaises(JointBudgetExceeded):
            p.solve(seconds=2, max_states=2)
        self.assertGreater(int(p.result['lower']), 0)
        self.assertEqual(p.tv_bound, Fraction(1, 2))
        with self.assertRaisesRegex(ValueError, 'uncertified draw'):
            p.native.draw_paths(1, 1)
        with self.assertRaisesRegex(ValueError, 'uncertified features'):
            p.native.revealed_context_masses()

    def test_manifest_permutation_is_rejected(self):
        model, manifest, table = fixture()
        manifest['species']['a'].reverse()
        with self.assertRaisesRegex(ValueError, 'binding'):
            NativeJointPosterior(model, manifest, table)

    def test_physical_manifest_tampering_is_rejected(self):
        from pokezero.generator_weights import GeneratorWeights, digest
        model, manifest, table = fixture()
        table = GeneratorWeights(table.provenance, table.source_hash, table.species, digest(manifest))
        manifest['species']['a'][0]['moves'] = ['tackle']
        with self.assertRaisesRegex(ValueError, 'physical manifest binding'):
            NativeJointPosterior(model, manifest, table)

    def test_reverse_preparation_failure_uses_stage1_boundary(self):
        with patch('pokezero.joint_native.validate_pinned_catalog'), patch(
                'pokezero.joint_native.battle_joint_parameters', side_effect=ImpossibleJointEvidence('own team')):
            result = draw_with_stage1_fallback(*fixture(), count=1, rng=random.Random(1),
                stage1_sampler=lambda count, rng: ('stage1',),
                battle_context=dict(owner='p2', own_initial_team=(), producer='shared-battle'))
        self.assertEqual(result.mode, 'stage1_fallback')
        self.assertEqual(result.teams, ('stage1',))

    def test_serving_requires_explicit_battle_binding(self):
        options = dict(count=1, rng=random.Random(1), stage1_sampler=lambda count, rng: ())
        with patch('pokezero.joint_native.validate_pinned_catalog'):
            with self.assertRaises(TypeError):
                draw_with_stage1_fallback(*fixture(), **options)
            for context in (None, {}, {'owner': 'p2'}):
                with self.assertRaisesRegex(ValueError, 'battle context'):
                    draw_with_stage1_fallback(*fixture(), **options, battle_context=context)

    def test_serving_rejects_self_consistent_replacement_catalog(self):
        from pathlib import Path
        from pokezero.generator_weights import digest
        from pokezero.joint_native import validate_pinned_catalog
        model = json.loads((Path(__file__).parents[1] / 'src/pokezero/data/gen3_team_model.json').read_text())
        model['bases'][0]['tokens'] += 1
        model['sha256'] = digest({k: v for k, v in model.items() if k != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'pinned team catalog'):
            validate_pinned_catalog(model, {}, None)

    def test_duplicate_own_physical_set_is_one_observation(self):
        from pokezero.generator_weights import GeneratorWeights, SpeciesWeights
        model, manifest, table = fixture()
        original = manifest['species']['c'][0]
        manifest['species']['c'].append(dict(original, variant_id='c-duplicate'))
        species = dict(table.species)
        species['c'] = SpeciesWeights(1, ('c-0', 'c-duplicate'), ((1, 0),)*16)
        table = GeneratorWeights('fixture', 'fixture', species)
        first = own_team_terminal_weights(model, manifest, table, (('a', 1), ('b', 1), ('c', 0)))
        duplicate = own_team_terminal_weights(model, manifest, table, (('a', 1), ('b', 1), ('c', 1)))
        self.assertEqual(first, duplicate)


if __name__ == '__main__':
    unittest.main()
