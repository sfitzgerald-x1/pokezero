import collections
import json
import math
import unittest

from gpu_engine import fp_reply_evidence as ev


def member(species, moves, ability, item):return dict(species=species, moves=moves, ability=ability, item=item)


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
        world = lambda g: dict(opponent=dict(sets=[member(*g), member(*SNORLAX)]))
        rr = [dict(ordinal=4, decision_index=7, worlds=[world(GENGAR_A), world(GENGAR_B)]),
              dict(ordinal=4, decision_index=7, worlds=[world(GENGAR_A), dict(opponent={})]),
              dict(ordinal=4, decision_index=8, worlds=[world(GENGAR_A)]),
              dict(ordinal=4, decision_index=7, worlds=None)]
        out, counts = ev.weights(rows, rr)
        self.assertEqual(out, [dict(ordinal=4, decision_index=7, world=0, loglik=-1.), dict(ordinal=4, decision_index=7, world=1, loglik=-3.)])
        self.assertEqual(dict(counts), dict(decisions=4, weighted=1, with_evidence=1, world_sets_missing=1, no_evidence_row=1, no_worlds=1))


if __name__ == '__main__':
    unittest.main()
