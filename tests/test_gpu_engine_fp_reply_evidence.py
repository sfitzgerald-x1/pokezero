import collections
import json
import math
import tempfile
import unittest
from pathlib import Path

from gpu_engine import fp_reply_evidence as ev


def member(species, moves, ability, item):return dict(species=species, moves=moves, base_ability=ability, ability='Trace copy', item=item)


GENGAR_A = ('Gengar', ['Thunderbolt', 'Ice Punch', 'Hypnosis', 'Explosion'], 'Levitate', 'Leftovers')
GENGAR_B = ('Gengar', ['Thunderbolt', 'Ice Punch', 'Giga Drain', 'Explosion'], 'Levitate', 'Leftovers')
SNORLAX = ('Snorlax', ['Body Slam', 'Return', 'Curse', 'Rest'], 'Immunity', 'Leftovers')


def evidence_of(*entries):
    """fp_reply_reweight's evidence tables: entries are (species id, [(set parts, loglik)])."""
    evidence = collections.defaultdict(dict);n = collections.Counter()
    for mon, scored in entries:
        for parts, value in scored:evidence[mon][ev.set_key(*parts[1:])] = value
        n[mon] = 3
    return evidence, n


class EvidenceTests(unittest.TestCase):
    def test_set_key_is_variant_key(self):
        variant = dict(moves=['Return 102', 'Hidden Power Fire 70'], ability='Thick Fat', item='Leftovers')
        self.assertEqual(ev.set_key(variant['moves'], variant['ability'], variant['item']), ev.rw.variant_key(variant))

    def test_snapshot_round_trips_and_keeps_minus_infinity_as_null(self):
        evidence, n = evidence_of(('gengar', [(GENGAR_A, -1.25), (GENGAR_B, -math.inf)]))
        evidence['snorlax'][ev.set_key(*SNORLAX[1:])] = -.5   # scored once but never public: no evidence count
        snap = json.loads(json.dumps(ev.snapshot(evidence, n, {'gengar'})))
        self.assertEqual(set(snap), {'gengar'});self.assertFalse(snap['gengar']['stats_aligned']);self.assertEqual(snap['gengar']['n'], 3)
        table = ev.table_of(snap)
        self.assertEqual(table, {'gengar': {ev.set_key(*GENGAR_A[1:]): -1.25, ev.set_key(*GENGAR_B[1:]): None}})

    def test_world_loglik(self):
        evidence, n = evidence_of(('gengar', [(GENGAR_A, -1.25), (GENGAR_B, -math.inf)]), ('snorlax', [(SNORLAX, -.5)]))
        table = ev.table_of(json.loads(json.dumps(ev.snapshot(evidence, n, set()))))
        sets = lambda *parts: {ev.rw.toid(p[0]): ev.set_key(*p[1:]) for p in parts}
        self.assertEqual(ev.world_loglik(table, sets(GENGAR_A, SNORLAX)), (-1.75, None))
        self.assertEqual(ev.world_loglik(table, sets(GENGAR_B, SNORLAX)), (None, 'set_ruled_out'))
        self.assertEqual(ev.world_loglik(table, sets(GENGAR_A)), (None, 'species_not_in_world'))
        other = ('Snorlax', ['Body Slam', 'Earthquake', 'Curse', 'Rest'], 'Immunity', 'Leftovers')
        self.assertEqual(ev.world_loglik(table, sets(GENGAR_A, other)), (None, 'set_not_scored'))
        self.assertEqual(ev.world_loglik({}, sets(GENGAR_A)), (0., None))

    def test_weights_join_every_world_or_none(self):
        evidence, n = evidence_of(('gengar', [(GENGAR_A, -1.), (GENGAR_B, -3.)]))
        rows = [dict(ordinal=4, decision_index=7, round=9, mons=ev.snapshot(evidence, n, set()))]
        world = lambda g: dict(opponent=dict(party=[member(*g), member(*SNORLAX)]))
        rr = [dict(ordinal=4, decision_index=7, worlds=[world(GENGAR_A), world(GENGAR_B)]),
              dict(ordinal=4, decision_index=7, worlds=[world(GENGAR_A), dict(opponent={})]),
              dict(ordinal=4, decision_index=8, worlds=[world(GENGAR_A)]),
              dict(ordinal=4, decision_index=7, worlds=None)]
        out, counts = ev.weights(rows, rr)
        self.assertEqual(out, [dict(ordinal=4, decision_index=7, world=0, loglik=-1.), dict(ordinal=4, decision_index=7, world=1, loglik=-3.)])
        self.assertEqual(dict(counts), dict(decisions=4, worlds=5, worlds_joined=3, worlds_weighted=2, weighted=1, with_evidence=1,
                                            worlds_failed_world_sets_missing=1, worlds_failed_no_evidence_row=1, decisions_not_weighted=2,
                                            no_worlds=1, members_keyed_by_current_ability=0))
        self.assertEqual(ev.coverage(counts), .6)
        out, counts = ev.weights(rows, rr[:1] + rr[3:], drop={4: 'replay diverges'})
        self.assertEqual((out, counts['dropped_battle_decisions'], counts['worlds'], ev.coverage(counts)), ([], 2, 0, None))

    def test_tracer_keys_on_base_ability_not_the_copy(self):
        gardevoir = ('Gardevoir', ['Psychic', 'Calm Mind', 'Fire Punch', 'Thunderbolt'], 'Trace', 'Leftovers')
        evidence, n = evidence_of(('gardevoir', [(gardevoir, -2.)]), ('gengar', [(GENGAR_A, -1.)]))
        rows = [dict(ordinal=1, decision_index=0, round=0, mons=ev.snapshot(evidence, n, set()))]
        evidence, n = evidence_of(('gengar', [(GENGAR_A, -1.)]))
        rows.append(dict(ordinal=1, decision_index=1, round=1, mons=ev.snapshot(evidence, n, set())))   # Gardevoir not yet public
        tracer = dict(species='Gardevoir', moves=gardevoir[1], base_ability='Trace', ability='Guts', item='Leftovers')
        world = dict(opponent=dict(party=[tracer, member(*GENGAR_A)]))
        out, counts = ev.weights(rows, [dict(ordinal=1, decision_index=0, worlds=[world]), dict(ordinal=1, decision_index=1, worlds=[world])])
        self.assertEqual([r['loglik'] for r in out], [-3., -1.]);self.assertEqual(ev.coverage(counts), 1.)
        self.assertEqual({k: counts[k] for k in ('tracer_worlds', 'tracer_worlds_with_evidence', 'tracer_worlds_joined',
                                                 'tracer_worlds_joined_with_evidence', 'tracer_members_copied_ability')},
                         dict(tracer_worlds=2, tracer_worlds_with_evidence=1, tracer_worlds_joined=2, tracer_worlds_joined_with_evidence=1,
                              tracer_members_copied_ability=2))
        copied_only = dict(species='Gardevoir', moves=gardevoir[1], ability='Guts', item='Leftovers')   # no base_ability: keyed on the copy
        out, counts = ev.weights(rows[:1], [dict(ordinal=1, decision_index=0, worlds=[dict(opponent=dict(party=[copied_only, member(*GENGAR_A)]))])])
        self.assertEqual((out, counts['worlds_failed_set_not_scored'], counts['members_keyed_by_current_ability'], counts['tracer_worlds']), ([], 1, 1, 0))

    def test_join_rules_unown_forms_removed_items_and_undecided_roots(self):
        unown = ('Unown-U', ['Hidden Power Psychic'], 'Levitate', 'Choice Specs')
        raichu_a = ('Raichu', ['Thunderbolt', 'Surf', 'Substitute', 'Hidden Power Ice'], 'Static', 'Petaya Berry')
        raichu_b = ('Raichu', ['Thunderbolt', 'Surf', 'Substitute', 'Hidden Power Ice'], 'Static', 'Leftovers')
        evidence, n = evidence_of(('unownu', [(unown, -.5)]), ('raichu', [(raichu_a, -1.)]))
        rows = [dict(ordinal=2, decision_index=0, round=0, mons=ev.snapshot(evidence, n, set()))]
        world = lambda raichu_item: dict(opponent=dict(party=[member('unown', unown[1] + [None] * 3, *unown[2:]),   # empty slots: None
                                                                member('Raichu', raichu_a[1], 'Static', raichu_item)]))
        undecided = dict(ordinal=2, decision_index=5, root=dict(decided=False), worlds=[dict(constructed=False)])
        out, counts = ev.weights(rows, [dict(ordinal=2, decision_index=0, root=dict(decided=True), worlds=[world('Petaya Berry'), world('')]), undecided])
        self.assertEqual([r['loglik'] for r in out], [-1.5, -1.5])
        self.assertEqual((counts['item_removed_resolved'], counts['undecided_decisions'], counts['worlds'], ev.coverage(counts)), (1, 1, 2, 1.))
        evidence, n = evidence_of(('raichu', [(raichu_a, -1.), (raichu_b, -2.)]))   # two sets differ only by item: '' can't choose
        rows = [dict(ordinal=2, decision_index=0, round=0, mons=ev.snapshot(evidence, n, set()))]
        out, counts = ev.weights(rows, [dict(ordinal=2, decision_index=0, worlds=[world('')])])
        self.assertEqual((out, counts['worlds_failed_item_removed_ambiguous']), ([], 1))
        self.assertEqual(ev.species_key('Unown-!'), 'unown');self.assertEqual(ev.species_key('Porygon2'), 'porygon2')

    def test_extra_sets_from_decided_worlds_with_items(self):
        aero = ('Aerodactyl', ['doubleedge', 'earthquake', 'hiddenpowerflying', 'rockslide'], 'rockhead', 'choiceband')
        consumed = ('Raichu', ['thunderbolt', 'surf', 'substitute', 'hiddenpowerice'], 'static', '')
        world = dict(opponent=dict(party=[member(*aero), member(*consumed), dict(species='Unown-U', moves=['hiddenpowerpsychic', None, None, None],
                                                                             base_ability='levitate', item='choicespecs')]))
        rr = [dict(ordinal=755, decision_index=11, root=dict(decided=True), worlds=[world, world]),
              dict(ordinal=755, decision_index=12, root=dict(decided=False), worlds=[dict(opponent=dict(party=[member('Gengar', *GENGAR_A[1:])]))])]
        extras = ev.extra_sets(rr)
        self.assertEqual(sorted(extras[755]), ['aerodactyl', 'unown'])
        self.assertEqual(list(extras[755]['aerodactyl'].values()), [dict(moves=aero[1], ability='rockhead', item='choiceband')])
        self.assertEqual(ev.rw.variant_key(next(iter(extras[755]['aerodactyl'].values()))), ev.set_key(*aero[1:]))
        self.assertEqual(list(extras[755]['unown'].values()), [dict(moves=['hiddenpowerpsychic'], ability='levitate', item='choicespecs')])

    def test_an_extra_set_skipped_at_a_round_is_incomplete_and_never_joins(self):
        evidence, n = evidence_of(('gengar', [(GENGAR_A, -1.), (GENGAR_B, -2.)]))
        evidence['gengar'][ev.set_key(*GENGAR_B[1:])] = float('nan')   # fp_reply_reweight marks a skipped extra set this way
        rows = [dict(ordinal=4, decision_index=7, round=0, mons=json.loads(json.dumps(ev.snapshot(evidence, n, set()))))]
        self.assertIn([sorted(ev.rw.move_key(m) for m in GENGAR_B[1]), 'levitate', 'leftovers', 'incomplete'], rows[0]['mons']['gengar']['sets'])
        world = lambda g: dict(opponent=dict(party=[member(*g)]))
        out, counts = ev.weights(rows, [dict(ordinal=4, decision_index=7, worlds=[world(GENGAR_A), world(GENGAR_B)])])
        self.assertEqual((out, counts['worlds_failed_set_incomplete'], counts['worlds_joined']), ([], 1, 1))

    def test_cli_writes_weights_only_at_full_coverage(self):
        evidence, n = evidence_of(('gengar', [(GENGAR_A, -1.)]))
        world = dict(opponent=dict(party=[member(*GENGAR_A)]))
        rr = [dict(ordinal=4, decision_index=7, worlds=[world]), dict(ordinal=9, decision_index=0, worlds=[world])]   # 9: no evidence row
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp);(tmp / 'ev').mkdir()
            (tmp / 'ev' / 'evidence-0.jsonl').write_text(json.dumps(dict(ordinal=4, decision_index=7, round=0, mons=ev.snapshot(evidence, n, set()))) + '\n')
            (tmp / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rr))
            (tmp / 'drop.json').write_text(json.dumps([dict(ordinal=9, reason='replay diverges')]))
            base = ['weights', '--evidence', str(tmp / 'ev'), '--rows', str(tmp / 'rows.jsonl')]
            self.assertEqual(ev.main(base + ['--out', str(tmp / 'a.jsonl')]), 1)
            self.assertFalse((tmp / 'a.jsonl').exists())
            summary = json.loads((tmp / 'a.jsonl.summary.json').read_text())
            self.assertEqual(summary['coverage'], dict(worlds=2, joined=1, share=.5, full=False, failed=dict(no_evidence_row=1)))
            self.assertEqual(ev.main(base + ['--out', str(tmp / 'b.jsonl'), '--allow-partial']), 0)
            self.assertEqual(len((tmp / 'b.jsonl').read_text().splitlines()), 1)
            self.assertEqual(ev.main(base + ['--out', str(tmp / 'c.jsonl'), '--drop-battles', str(tmp / 'drop.json')]), 0)
            summary = json.loads((tmp / 'c.jsonl.summary.json').read_text())
            self.assertEqual((summary['coverage']['full'], summary['counts']['dropped_battle_decisions'], summary['dropped_battles']),
                             (True, 1, {'9': 'replay diverges'}))



class TrainingSeedsTests(unittest.TestCase):
    def test_reads_every_shards_seed_array_only(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            np.savez(d / 'shard-0.npz', seed=np.array([5, 7, 5]), label=np.array([1, 2, 3]), numeric_features=np.zeros((3, 2)))
            np.savez(d / 'shard-1.npz', seed=np.array([9], np.int64), label=np.array([0]))
            np.savez(d / 'shard-2.npz', seed=np.array([], np.int64), label=np.array([], np.int64))
            np.savez(d / 'other.npz', seed=np.array([11]))   # not a shard
            self.assertEqual(ev.training_seeds(d), {5, 7, 9})

    def test_with_the_split_only_fitted_games_count(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)   # seeds 20 test (fixed), 31 validation (% 10 == 1), 42 train, 53 late (first ordinal >= 100)
            np.savez(d / 'shard-0.npz', seed=np.array([20, 31, 42, 42, 53]), ordinal=np.array([1, 2, 3, 3, 100]), label=np.zeros(5))
            np.savez(d / 'shard-1.npz', seed=np.array([53, 64]), ordinal=np.array([101, 4]), label=np.zeros(2))
            self.assertEqual(ev.training_seeds(d, [20], 100), {31, 42, 64})
            self.assertEqual(ev.training_seeds(d, [20]), {31, 42, 53, 64})
            with self.assertRaises(ValueError):ev.training_seeds(d, None, 100)


if __name__ == '__main__':
    unittest.main()
