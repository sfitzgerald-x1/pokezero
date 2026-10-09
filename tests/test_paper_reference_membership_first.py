"""Necessary-party early rejection preserves the original accepted joint law."""
from collections import Counter
import hashlib
import random
from types import SimpleNamespace
import unittest

from pokezero.mcts_eval.paper_reference import ReferenceRefusal, SamplingDeadlineExceeded
from pokezero.mcts_eval.paper_reference_sampling import KnownSetTraits, PaperHiddenTeamSampler
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
from pokezero.showdown_fixture import pack_team
from tests.test_paper_reference_sampler import Generator, set_row
from _showdown_root import requires_showdown, showdown_root
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.randbat import load_gen3_randbat_source_cached, canonical_gen3_randbat_species_id
from pokezero.mcts_eval.paper_reference_sampling import _fixture


def receipt():
    return dict(complete_proposals=0, matches=0, membership_rejections=0,
                known_completions_materialized=0)


def sampler(generator):
    return PaperHiddenTeamSampler(generator,
        set_source=SimpleNamespace(species_metadata={}, move_metadata={}))


class MembershipFirstTests(unittest.TestCase):
    def test_default_seed_coupling_still_draws_known_then_unknown(self):
        generator = Generator()
        rng = random.Random(11)
        draw = sampler(generator).draw((KnownSetTraits('Snorlax'),), rng)
        expected = random.Random(11)
        self.assertEqual(generator.calls, [('set', expected.getrandbits(32), 'Snorlax'),
                                          ('party', expected.getrandbits(32))])
        self.assertEqual(rng.getstate(), expected.getstate())
        self.assertEqual(draw.known[0].seeds, (generator.calls[0][1],))

    def test_rejected_unknown_parties_do_not_generate_known_sets(self):
        class Sequence(Generator):
            def generate_scenario_team(self, *, seed):
                count = sum(c[0] == 'party' for c in self.calls)
                self.calls.append(('party', seed))
                species = ('Murkrow' if count == 4 else 'Pikachu', 'Charizard',
                           'Blastoise', 'Venusaur', 'Gengar', 'Alakazam')
                return [set_row(s) for s in species]
        g, evidence = Sequence(), receipt()
        draw = sampler(g).draw_membership_first((KnownSetTraits('Snorlax'),), random.Random(3),
            required=frozenset({'murkrow'}), check=lambda: None, receipt=evidence)
        self.assertEqual([c[0] for c in g.calls], ['party']*5+['set'])
        self.assertEqual(evidence, dict(complete_proposals=5, matches=1,
            membership_rejections=4, known_completions_materialized=1))
        self.assertEqual([m.species for m in draw.team],
            ['Snorlax','Murkrow','Charizard','Blastoise','Venusaur','Gengar'])
        self.assertEqual(draw.unknown_party_seeds, (g.calls[-2][1],))
        self.assertEqual(draw.packed_team_sha256, hashlib.sha256(pack_team(draw.team).encode()).hexdigest())

    def test_membership_checks_retained_completion_not_discarded_sixth_mon(self):
        g = Generator(party=[set_row(s) for s in
            ('Pikachu','Charizard','Blastoise','Venusaur','Gengar','Murkrow')])
        evidence = receipt()
        with self.assertRaisesRegex(ReferenceRefusal, 'membership exhausted'):
            sampler(g).draw_membership_first((KnownSetTraits('Snorlax'),), random.Random(3),
                required=frozenset({'murkrow'}), check=lambda: None, receipt=evidence)
        self.assertEqual(evidence['membership_rejections'], 2048)
        self.assertFalse(any(c[0] == 'set' for c in g.calls))

    def test_fixed_known_species_already_satisfies_predicate_without_forcing_unknowns(self):
        g, evidence = Generator(), receipt()
        draw = sampler(g).draw_membership_first((KnownSetTraits('Snorlax'),), random.Random(3),
            required=frozenset({'snorlax'}), check=lambda: None, receipt=evidence)
        self.assertEqual(evidence['complete_proposals'], 1)
        self.assertEqual([c[0] for c in g.calls], ['party','set'])
        self.assertEqual(len({m.species for m in draw.team}), 6)

    def test_native_known_error_still_propagates_after_membership_acceptance(self):
        class Broken(Generator):
            def generate_reference_set(self, **kwargs):
                raise RuntimeError('native known draw failed')
        evidence = receipt()
        with self.assertRaisesRegex(RuntimeError, 'native known draw failed'):
            sampler(Broken()).draw_membership_first((KnownSetTraits('Snorlax'),), random.Random(3),
                required=frozenset({'snorlax'}), check=lambda: None, receipt=evidence)
        self.assertEqual(evidence['matches'], 1)
        self.assertEqual(evidence['known_completions_materialized'], 0)

    def test_native_party_error_is_not_interpreted_as_membership_rejection(self):
        class Broken(Generator):
            def generate_scenario_team(self, **kwargs):
                raise RuntimeError('native party draw failed')
        evidence = receipt()
        with self.assertRaisesRegex(RuntimeError, 'native party draw failed'):
            sampler(Broken()).draw_membership_first((KnownSetTraits('Snorlax'),), random.Random(3),
                required=frozenset({'snorlax'}), check=lambda: None, receipt=evidence)
        self.assertEqual(evidence['complete_proposals'], 0)

    def test_deadline_after_complete_unknown_draw_does_not_materialize_or_use_it(self):
        calls, evidence, g = 0, receipt(), Generator()
        def check():
            nonlocal calls
            calls += 1
            if calls == 2: raise SamplingDeadlineExceeded('deadline')
        with self.assertRaises(SamplingDeadlineExceeded):
            sampler(g).draw_membership_first((KnownSetTraits('Snorlax'),), random.Random(3),
                required=frozenset({'snorlax'}), check=check, receipt=evidence)
        self.assertEqual(evidence['complete_proposals'], 1)
        self.assertEqual(evidence['matches'], 0)
        self.assertFalse(any(c[0] == 'set' for c in g.calls))

    def test_original_multi_party_safety_cap_is_preserved(self):
        g = Generator(party=[set_row('Snorlax')]*6)
        with self.assertRaisesRegex(ReferenceRefusal, 'fresh-party rejection safety cap'):
            sampler(g).draw_membership_first((KnownSetTraits('Snorlax'),), random.Random(3),
                required=frozenset({'snorlax'}), check=lambda: None, receipt=receipt())
        self.assertEqual(len(g.calls), 10)

    def test_invalid_known_inputs_refuse_before_any_generator_calls(self):
        g = Generator()
        with self.assertRaisesRegex(ReferenceRefusal, 'species clause'):
            sampler(g).draw_membership_first((KnownSetTraits('Snorlax'),)*2, random.Random(3),
                required=frozenset({'snorlax'}), check=lambda: None, receipt=receipt())
        self.assertEqual(g.calls, [])

    def test_finite_target_oracle_and_empirical_joint_distribution(self):
        # Known draw A/B = 1/2; four unknown-party types = 1/4 each.
        # Membership accepts two types. Conditioning preserves A/B=1/2 and
        # both accepted full-context party types=1/2, giving four joints=1/4.
        original_mass = {(k,u): .5*.25 for k in range(2) for u in range(4)}
        accepted = {key: mass for key,mass in original_mass.items() if key[1] < 2}
        z = sum(accepted.values())
        self.assertEqual({key: mass/z for key,mass in accepted.items()},
                         {(k,u): .25 for k in range(2) for u in range(2)})
        class Bits(Generator):
            def generate_reference_set(self, *, seed, species):
                return set_row('Snorlax', item='Leftovers' if seed&1 else 'Choice Band')
            def generate_scenario_team(self, *, seed):
                kind = seed % 4
                species = ('Murkrow' if kind<2 else 'Pikachu',
                           'Charizard' if kind%2 else 'Blastoise',
                           'Venusaur','Gengar','Alakazam','Zapdos')
                return [set_row(s) for s in species]
        for early in (False, True):
            rng, counts, s = random.Random(9), Counter(), sampler(Bits())
            for _ in range(2048):
                if early:
                    draw=s.draw_membership_first((KnownSetTraits('Snorlax'),), rng,
                        required=frozenset({'murkrow'}), check=lambda:None, receipt=receipt())
                else:
                    while True:
                        draw=s.draw((KnownSetTraits('Snorlax'),),rng)
                        if draw.team[1].species=='Murkrow':break
                counts[draw.team[0].item,draw.team[2].species]+=1
            self.assertEqual(len(counts),4)
            for count in counts.values():self.assertAlmostEqual(count/2048,.25,delta=.035)

    def test_runtime_opt_in_is_default_off_and_requires_particles(self):
        self.assertFalse(ShowdownWorkerFactory('c','d','e','s').membership_first)
        for bad in (True,1,'true'):
            with self.assertRaisesRegex(ReferenceRefusal,'membership-first'):
                ShowdownWorkerFactory('c','d','e','s',membership_first=bad)
        self.assertTrue(ShowdownWorkerFactory('c','d','e','s',history_particles=32,
                                            membership_first=True).membership_first)

    @requires_showdown()
    def test_native_unknown_party_context_and_known_set_receipts_are_unchanged(self):
        env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root()))
        try:
            source = load_gen3_randbat_source_cached(showdown_root())
            s = PaperHiddenTeamSampler(env,set_source=source)
            # Fixed predicate needs no outcome-selected fixture. Verify both
            # independent native kernels from THEIR retained original seeds.
            for seed in (7,19,93):
                evidence=receipt()
                draw=s.draw_membership_first((KnownSetTraits('Snorlax'),),random.Random(seed),
                    required=frozenset({'snorlax'}),check=lambda:None,receipt=evidence)
                expected=[];seen={'snorlax'}
                for party_seed in draw.unknown_party_seeds:
                    for row in env.generate_scenario_team(seed=party_seed):
                        mon=_fixture(row);species=canonical_gen3_randbat_species_id(mon.species)
                        if species not in seen:seen.add(species);expected.append(mon)
                        if len(expected)==5:break
                self.assertEqual(tuple(expected),draw.team[1:])
                native=_fixture(env.generate_reference_set(seed=draw.known[0].seeds[0],species='Snorlax'))
                self.assertEqual(native,draw.team[0])
                self.assertEqual(evidence['known_completions_materialized'],1)
                self.assertEqual(len({canonical_gen3_randbat_species_id(m.species) for m in draw.team}),6)
        finally:env.close()


if __name__=='__main__':unittest.main()
