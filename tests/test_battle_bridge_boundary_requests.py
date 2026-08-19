"""Pins for preserving a streamed decision boundary in generic snapshots."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.showdown_fixture import FixturePokemon, pack_team

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "battle_bridge_boundary_requests.mjs"


def _select(stream: object, stream_is_current: bool, simulator: object) -> object:
    program = """
const { snapshotBoundaryRequests } = await import(process.argv[1]);
const [stream, streamIsCurrent, simulator] = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(snapshotBoundaryRequests(stream, streamIsCurrent, simulator)));
"""
    completed = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            program,
            str(HELPER),
            json.dumps([stream, stream_is_current, simulator]),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return json.loads(completed.stdout)


def _invalidate_and_select(stream: object, generation: int) -> tuple[dict[str, object], object]:
    program = """
const { invalidatedBoundaryState, snapshotBoundaryRequests } = await import(process.argv[1]);
const [stream, generation] = JSON.parse(process.argv[2]);
const state = invalidatedBoundaryState(generation);
const selected = snapshotBoundaryRequests(
  stream,
  state.boundaryRequestGeneration === state.boundaryGeneration,
  {},
);
process.stdout.write(JSON.stringify([state, selected]));
"""
    completed = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            program,
            str(HELPER),
            json.dumps([stream, generation]),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    state, selected = json.loads(completed.stdout)
    assert isinstance(state, dict)
    return state, selected


class SnapshotBoundaryRequestsTest(unittest.TestCase):
    def test_empty_direct_requests_preserve_streamed_actionable_boundary(self) -> None:
        streamed = {"p1": {"side": {"id": "p1"}, "active": [{}]}}

        self.assertEqual(_select(streamed, True, {}), streamed)

    def test_nonempty_direct_requests_replace_prior_stream_boundary(self) -> None:
        streamed = {
            "p1": {"side": {"id": "p1"}, "active": [{}]},
            "p2": {"side": {"id": "p2"}, "active": [{}]},
        }
        direct = {"p2": {"side": {"id": "p2"}, "forceSwitch": [True]}}

        self.assertEqual(_select(streamed, False, direct), direct)

    def test_stale_streamed_boundary_is_not_paired_with_a_new_snapshot(self) -> None:
        stale = {"p1": {"side": {"id": "p1"}, "active": [{}]}}

        state, selected = _invalidate_and_select(stale, generation=9)

        self.assertEqual(state, {
            "boundaryRequests": {},
            "boundaryGeneration": 10,
            "boundaryRequestGeneration": None,
        })
        self.assertEqual(selected, {})

    @requires_showdown("requires a built Pokemon Showdown checkout")
    def test_generic_snapshot_restores_an_actionable_boundary_in_a_fresh_shell(self) -> None:
        config = LocalShowdownConfig(showdown_root=showdown_root())
        start_override = BattleStartOverride(
            player_teams={
                "p1": pack_team(
                    (FixturePokemon(species="Charmander", ability="Blaze", moves=("Ember", "Tackle")),)
                ),
                "p2": pack_team(
                    (FixturePokemon(species="Squirtle", ability="Torrent", moves=("Water Gun", "Tackle")),)
                ),
            }
        )

        with LocalShowdownEnv(config) as source, LocalShowdownEnv(config) as restored:
            source.reset_with_start_override(seed=17, start_override=start_override)
            snapshot = source.snapshot_actionable_boundary()
            restored.reset_with_start_override(seed=19, start_override=start_override)
            restored.restore(snapshot)
            self.assertEqual(restored.requested_players(), ("p1", "p2"))
            result = restored.step({"p1": 0, "p2": 1})

        self.assertFalse(result.terminal)
