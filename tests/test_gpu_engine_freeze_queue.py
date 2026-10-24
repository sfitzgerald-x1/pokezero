"""Independent authored Showdown boundary; not full-state/pool qualification."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE = os.environ.get("NODE") or shutil.which("node")
SHOWDOWN = os.environ.get("POKEZERO_SHOWDOWN_ROOT")


@unittest.skipUnless(NODE and SHOWDOWN, "requires pinned Gen3 Showdown and Node")
class FreezeQueueOracleTests(unittest.TestCase):
    def test_new_freeze_still_gets_before_move_thaw_and_lum_cure(self):
        run = subprocess.run([NODE, str(Path(__file__).with_name("gpu_freeze_queue_oracle.cjs")),
                              SHOWDOWN], capture_output=True, text=True, check=True)
        rows = json.loads(run.stdout)
        for name in ("fresh", "fresh_other_seat", "preexisting_thaw", "fresh_lum"):
            with self.subTest(name=name):
                self.assertEqual((rows[name]["status"], rows[name]["pp"]), ("", 7))
                self.assertTrue(any("|Splash|" in event for event in rows[name]["events"]))
        for name in ("fresh_stays", "fresh_stays_other_seat", "preexisting_stays"):
            self.assertEqual((rows[name]["status"], rows[name]["pp"]), ("frz", 8))
            self.assertTrue(any("|cant|" in event for event in rows[name]["events"]))
        for name in ("fresh", "fresh_other_seat", "fresh_lum", "fresh_stays"):
            self.assertTrue(any("|-status|" in event and "|frz" in event
                                for event in rows[name]["events"]), name)
        self.assertEqual(rows["fresh_lum"]["item"], "")
        for name, row in rows.items():
            self.assertEqual(row["thawCalls"], 0 if name == "fresh_lum" else 1, name)


if __name__ == "__main__":
    unittest.main()
