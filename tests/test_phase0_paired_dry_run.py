import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pokezero.mcts_eval import fullgame_strength as reader

SPEC = importlib.util.spec_from_file_location("dry_run", ROOT / "scripts/phase0_paired_dry_run.py")
dry_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dry_run)


def row(*, seed=100, status="PUBLIC_WIN", capped=False, tied=False, score=1.0):
    return dict(seed=seed, battle_id=f"battle-gen3randombattle-controlled-{seed}",
                decision_rounds=250 if capped else 20, capped=capped,
                tied=tied, winner=None if capped or tied else "PokeZeroBot", pokezero_score=score,
                pokezero_won=not (capped or tied),
                terminal_witness=dict(schema="pokezero.controlled-terminal-witness.v1",
                    status=status, capped=capped, decision_rounds=250 if capped else 20,
                    max_decision_rounds=250, players={"p1": "PokeZeroBot", "p2": "FoulPlayBot"},
                    winner_player=None if capped or tied else "p1",
                    public_terminal_line=None if capped else "|tie" if tied else "|win|PokeZeroBot"))


class DryRunTests(unittest.TestCase):
    def test_named_win_is_seat_relative(self):
        self.assertEqual(dry_run.checked_outcome(reader, row(), "p1", round_cap=250).bounds(), (1, 1))
        self.assertFalse(dry_run.checked_outcome(reader, row(), "p2", round_cap=250).validated)

    def test_cap_legacy_half_point_remains_unknown(self):
        outcome = dry_run.checked_outcome(reader, row(status="ROUND_CAP", capped=True, score=.5), "p1", round_cap=250)
        self.assertEqual((outcome.status, outcome.bounds()), ("CAPPED", (0, 1)))

    def test_actual_tie_is_half_but_tier_is_not(self):
        tie = row(status="PUBLIC_TIE", tied=True, score=.5)
        self.assertEqual(dry_run.checked_outcome(reader, tie, "p1", round_cap=250).bounds(), (.5, .5))
        tie["terminal_witness"]["public_terminal_line"] = "|tier|gen3randombattle"
        self.assertFalse(dry_run.checked_outcome(reader, tie, "p1", round_cap=250).validated)

    def test_conflicting_or_boolean_metadata_never_validates(self):
        for key, value in [("pokezero_score", True), ("decision_rounds", True), ("capped", 0), ("winner", "other")]:
            with self.subTest(key=key):
                sample = row(); sample[key] = value
                self.assertEqual(dry_run.checked_outcome(reader, sample, "p1", round_cap=250).bounds(), (0, 1))

    def test_partial_roster_keeps_all512_cells_and_never_enables_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = dict(seed_start=100, seed_count=128, run_directory=tmp,
                         common_arguments=["--max-decision-rounds", "250"],
                         shards=[dict(name=a+"-"+s, seat=s) for a in ("candidate", "raw") for s in ("p1", "p2")])
            doc = dict(completed_games=1, pokezero_player="p1", opponent_policy_id="foul-play",
                       policy_mode="engine-mcts", game_results=[row()])
            path = Path(tmp) / "candidate-p1.json"
            path.write_text(json.dumps(doc))
            result = dry_run.read_panel(entry, reader)
            self.assertEqual((result["planned_cells"], result["recorded_cells"]), (512, 1))
            self.assertIsNone(result["result"]["primary_test"])
            self.assertFalse(result["result"]["screening_rule_passed"])
            doc["game_results"][0]["seed"] = 101
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "chronological prefix"):
                dry_run.read_panel(entry, reader)

    def test_conflicting_identity_battle_and_won_fields_remain_unknown(self):
        for key, value in [("battle_id", "battle-gen3randombattle-controlled-101"),
                           ("pokezero_won", False), ("pokezero_won", 1)]:
            with self.subTest(key=key, value=value):
                sample = row(); sample[key] = value
                self.assertFalse(dry_run.checked_outcome(reader, sample, "p1", round_cap=250).validated)
        sample = row()
        sample["terminal_witness"]["players"] = {"p1": "FoulPlayBot", "p2": "PokeZeroBot"}
        sample["terminal_witness"]["winner_player"] = "p2"
        self.assertFalse(dry_run.checked_outcome(reader, sample, "p1", round_cap=250).validated)

    def test_hash_labels_the_same_entry_bytes_that_were_parsed(self):
        frozen = ROOT / "src/pokezero/mcts_eval/fullgame_strength.py"
        original = json.dumps(dict(runtime=str(ROOT), analysis=dict(reader=dict(
            path="src/pokezero/mcts_eval/fullgame_strength.py",
            sha256=hashlib.sha256(frozen.read_bytes()).hexdigest())))).encode()
        entry_path = Path("/entry-snapshot.json")
        read_bytes = Path.read_bytes
        entry_reads = []
        def changing_entry(path):
            if path == entry_path:
                entry_reads.append(path)
                return original if len(entry_reads) == 1 else b"changed bytes"
            return read_bytes(path)
        output = io.StringIO()
        with patch.object(sys, "argv", ["dry-run", "--entry", str(entry_path)]), \
                patch.object(Path, "read_bytes", changing_entry), \
                patch.object(dry_run, "read_panel", return_value={}), \
                patch("sys.stdout", output):
            dry_run.main()
        self.assertEqual(len(entry_reads), 1)
        self.assertEqual(json.loads(output.getvalue())["entry_sha256"], hashlib.sha256(original).hexdigest())


if __name__ == "__main__":
    unittest.main()
