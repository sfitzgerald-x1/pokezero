#!/usr/bin/env python
"""`limit:world_sample_drag_target` must be a LAST RESORT, not a short-circuit.

WHY. This label used to be returned on the mere presence of a `|drag|` line, before
any component test ran. It therefore attached to every row that merely CO-OCCURRED
with a phaze — and it masked a real renderer defect on eleven holdout rows and one
dev row, where the engine dragged the *correct* Pokémon and only the Spikes chip's
tag differed. #1081 fixed the renderer and all twelve closed.

The cost of the short-circuit was not the label; it was that two reports concluded
those rows were comparison limits. `limit:` is the one disposition that ends
inquiry, and the C116 M1 rule now requires a written demonstration before it can be
used. A classifier that hands the label out on a protocol keyword cannot satisfy
that rule. reports/c117, reports/c118.

The class is still legitimate in its honest form — a phaze genuinely can put the
engine's branch set against Showdown's single realised sample — so it is kept, and
demoted to the residue after every component test has declined.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import engine_transition_differential as etd  # noqa: E402

DRAG_STEP = [
    "|move|p2a: Skarmory|Whirlwind|p1a: Sableye",
    "|-damage|p1a: Sableye|210/239|[from] Spikes",
    "|drag|p1a: Shuckle|Shuckle, L98, F|172/198 slp",
    "|-damage|p1a: Shuckle|148/198 slp|[from] Spikes",
    "|-heal|p1a: Shuckle|160/198 slp|[from] item: Leftovers",
    "|upkeep",
]


class DragLimitIsALastResort(unittest.TestCase):
    def test_a_component_mismatch_beats_the_drag_label(self):
        """The regression pin. A row with a NAMED component disagreement must be
        classified by that disagreement, even though a `|drag|` line is present.
        Under the old short-circuit this returned the drag limit and the real
        defect stayed invisible for eleven holdout rows."""
        cls = etd.classify_divergence(
            DRAG_STEP,
            # The real miss format, taken from _MISS_COMPONENTS_RE /
            # _MISS_SOURCE_RE rather than invented -- my first version of this
            # fixture used prose the classifier does not parse, so it fell through
            # to the drag check and the pin failed for the wrong reason.
            ["p1 observed_only=[('spikes', -24)] engine_only=[]"],
        )
        self.assertNotEqual(
            cls, "limit:world_sample_drag_target",
            "a row whose components disagree must not be filed as a drag-target "
            "limit merely because a |drag| line is present — that is how a "
            "renderer defect masqueraded as a comparison limit for 12 rows",
        )
        self.assertIn(
            "component", cls,
            f"expected a component-based class for a component miss, got {cls!r}",
        )

    def test_the_class_survives_for_a_row_nothing_else_explains(self):
        """The control, and the reason the class is kept rather than deleted. With
        no parsable component disagreement, a phaze row still lands on the drag
        limit — which is its honest form. If this ever fails, the demotion went too
        far and the class has become unreachable."""
        cls = etd.classify_divergence(DRAG_STEP, [])
        self.assertEqual(cls, "limit:world_sample_drag_target", cls)

    def test_a_non_drag_row_never_gets_the_drag_label(self):
        """Second control: the label must track the `|drag|` line, not the absence
        of a component miss."""
        cls = etd.classify_divergence(
            ["|move|p1a: Gligar|Rock Slide|p2a: Fearow", "|upkeep"], []
        )
        self.assertNotEqual(cls, "limit:world_sample_drag_target", cls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
