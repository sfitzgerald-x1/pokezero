"""Fixed/explicit gender and chi-square against pinned randomTeam construction."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
import unittest

from pokezero.determinization import _sample_gender
from tests._showdown_root import showdown_root


class WorldGenderTests(unittest.TestCase):
    def test_fixed_species_do_not_consume_a_draw(self):
        for gender in ('M', 'F', 'N'):
            source = SimpleNamespace(species_metadata={'example': {'gender': gender}})
            rng = random.Random(27);before = rng.getstate()
            self.assertEqual(_sample_gender('Example', set_source=source, rng=rng), gender)
            self.assertEqual(rng.getstate(), before)

    def test_chi_square_against_actual_generator_and_constructor(self):
        root = showdown_root()
        commit = subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'], text=True).strip()
        self.assertTrue(commit.startswith('f76228a1'), commit)
        rows = json.loads(subprocess.check_output([os.environ.get('NODE', 'node'),
            str(Path(__file__).with_name('world_gender_reference.cjs')), root], text=True, timeout=180))
        source = SimpleNamespace(species_metadata={name: row['metadata'] for name, row in rows.items()})
        rng = random.Random(61381)
        groups = defaultdict(lambda: [Counter(), Counter()])
        for name, row in rows.items():
            n = sum(row[g] for g in ('M', 'F', 'N'))
            sampled = Counter(_sample_gender(name, set_source=source, rng=rng) for _ in range(n))
            fixed = row['metadata'].get('gender')
            if fixed in ('M', 'F', 'N'):
                self.assertEqual(sampled, {fixed: n});self.assertEqual(row[fixed], n)
            else:
                self.assertEqual(row['N'], 0);self.assertEqual(sampled['N'], 0)
                group = str(row['metadata']['genderRatio'])
                groups[group][0].update({g: row[g] for g in ('M', 'F')})
                groups[group][1].update(sampled)
        self.assertGreaterEqual(len(groups), 3)
        statistics = {}
        for group, (reference, sampled) in groups.items():
            total = sum(reference.values())
            self.assertGreater(total, 20)
            # Equal-sized two-sample Pearson homogeneity test, df=1.
            statistic = sum((reference[g]-sampled[g])**2/(reference[g]+sampled[g]) for g in ('M', 'F'))
            statistics[group] = dict(n=total, reference=dict(reference), sampled=dict(sampled), chi_square=statistic)
            self.assertLess(statistic, 16.27, statistics[group])  # p > .000055, fixed before sampling
            self.assertLess(sum((sampled[g]-total/2)**2/(total/2) for g in ('M', 'F')), 16.27)
        print('WORLD_GENDER_CHI_SQUARE ' + json.dumps(statistics, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
