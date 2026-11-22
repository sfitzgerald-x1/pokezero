"""Conditional diagnostic facts are bound identically across Python and Node."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location('joint_check', ROOT / 'scripts/gen3_joint_chi_square.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class ConditionalDiagnosticBinding(unittest.TestCase):
    def node(self, facts):
        return subprocess.run(['node', '-e',
            "const m=require('./scripts/gen3_joint_reference.cjs');console.log(m.conditionDigest(JSON.parse(process.argv[1])))",
            json.dumps(facts)], cwd=ROOT, capture_output=True, text=True, timeout=10)

    def test_facts_hash_changes_and_matches_node(self):
        facts = dict(evidence={'b': [1, 0], 'a': [0, 1]}, fixed_slots={'0': 'a'}, initial_wobbuffet=False)
        first = CHECK.condition_digest(facts)
        self.assertEqual(self.node(facts).stdout.strip(), first)
        facts['initial_wobbuffet'] = True
        self.assertNotEqual(CHECK.condition_digest(facts), first)
        self.assertEqual(self.node(facts).stdout.strip(), CHECK.condition_digest(facts))
        second = CHECK.condition_digest(facts)
        facts['fixed_slots'] = {'1': 'a'}
        self.assertNotEqual(CHECK.condition_digest(facts), second)
        facts['evidence']['b'] = [0, 1]
        self.assertEqual(self.node(facts).stdout.strip(), CHECK.condition_digest(facts))

    def test_nonunit_likelihood_is_not_an_event_probability(self):
        for row in ([2, 2, 0], [True, False], [-1, 1]):
            facts = dict(evidence={'a': row}, fixed_slots={}, initial_wobbuffet=False)
            with self.assertRaisesRegex(ValueError, 'Boolean evidence'):
                CHECK.condition_digest(facts)
            self.assertNotEqual(self.node(facts).returncode, 0)

    def test_catalog_regenerates_from_pinned_source(self):
        from tests._showdown_root import showdown_root
        root = os.environ.get('POKEZERO_SHOWDOWN_ROOT') or str(showdown_root())
        result = subprocess.run(['node', '-e',
            "const m=require('./scripts/export_gen3_team_model.cjs');console.log(JSON.stringify(m.exportModel(process.argv[1])))",
            root], cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
        self.assertEqual(json.loads(result.stdout),
                         json.loads((ROOT / 'src/pokezero/data/gen3_team_model.json').read_text()))
