"""Keep the pull request merge gate within its explicitly approved scope."""

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PR_EVENTS = {"pull_request", "pull_request_target"}


def pr_events(workflow):
    events = workflow.get("on", [])
    if isinstance(events, str):
        events = [events]
    return sorted(PR_EVENTS.intersection(events))


class PremergePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        github = ROOT / ".github"
        cls.allowlist = json.loads((github / "ci-premerge-allowlist.json").read_text())
        # BaseLoader keeps the literal Actions key "on" instead of coercing it to True.
        cls.workflows = {
            path.name: yaml.load(path.read_text(), Loader=yaml.BaseLoader)
            for path in (github / "workflows").iterdir()
            if path.suffix in {".yml", ".yaml"}
        }

    def test_pr_workflow_triggers_match_allowlist(self):
        active = {
            name: events
            for name, workflow in self.workflows.items()
            if (events := pr_events(workflow))
        }
        self.assertEqual(active, self.allowlist["pr_workflows"])

    def test_premerge_jobs_and_lanes_match_allowlist(self):
        jobs = self.workflows["premerge.yml"]["jobs"]
        self.assertCountEqual(jobs, self.allowlist["premerge_jobs"])
        core = jobs["core"]
        self.assertEqual(core["timeout-minutes"], "20")
        matrix = core["strategy"]["matrix"]
        self.assertEqual(set(matrix), {"include"})
        self.assertCountEqual(
            [entry["lane"] for entry in matrix["include"]],
            self.allowlist["core_lanes"],
        )
