"""ACCEPTANCE: forcing any of the four harnesses to skip 100% of boundaries exits nonzero.

Before this rule, each of the four could report a pass while measuring nothing:

- ``leaf_vs_reality.py`` gated on ``state``/``turn`` divergences, which are zero when nothing
  is compared. It really did print ``DEFECT-CLASS divergent boundaries: 0`` through a run that
  skipped 100% of boundaries on a schema-guard bug (C112).
- ``leaf_root_parity.py`` gated on ``all(divergent == 0)`` and ``prior_mapping_assert.py`` on
  ``all(mismatch == 0)`` — both **vacuously true over zero rows**.
- ``fidelity_gate_events.py`` ended in an unconditional ``return 0``. It could not fail.

Each test below drives the harness's real ``main()`` with ``run_corpus`` replaced by a report
describing a fully-skipped corpus, which is exactly the condition the acceptance names. The
paired ``*_still_passes_when_it_measured_something`` test is the non-vacuity control: without
it, a harness that returned 1 unconditionally would satisfy the acceptance while being useless.

Simulating at ``run_corpus`` rather than at each harness's internal driver is deliberate — it
is the one seam the four share, so the four tests exercise the same contract rather than four
different mocks, and it tests the wiring in ``main()``, which is what the acceptance is about.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

#: (module, the report a FULLY-SKIPPED corpus produces, the report a healthy one produces).
#: The skipped shape is per-harness because their counter vocabularies differ; what they share
#: is `attempted == 0`, which is the condition the rule fires on.
HARNESSES = [
    (
        "leaf_vs_reality",
        {
            "corpus": "forced",
            "boundaries": 1008,
            "counts": {"skip:forced": 1008, "attempted": 0},
            "class_rows": {},
            "families": [],
            "matchup_divergent_boundaries": 0,
            "live_tendency_divergent_boundaries": 0,
            "matchup_excess": [],
        },
        {
            "corpus": "healthy",
            "boundaries": 10,
            "counts": {"attempted": 10, "exact": 10, "divergent": 0},
            "class_rows": {},
            "families": [],
            "matchup_divergent_boundaries": 0,
            "live_tendency_divergent_boundaries": 0,
            "matchup_excess": [],
        },
    ),
    (
        "leaf_root_parity",
        {"corpus": "forced", "rows": 500, "counts": {"skip:forced": 500, "attempted": 0}, "families": []},
        {
            "corpus": "healthy",
            "rows": 10,
            "counts": {"attempted": 10, "exact": 10, "divergent": 0},
            "families": [],
        },
    ),
    (
        "prior_mapping_assert",
        {
            "corpus": "forced",
            "rows": 500,
            "root_narrower_rows": 0,
            "counts": {"skip:forced": 500, "attempted": 0},
            "mismatches": [],
        },
        {
            "corpus": "healthy",
            "rows": 10,
            "root_narrower_rows": 0,
            "counts": {"attempted": 10, "exact": 10, "mismatch": 0},
            "mismatches": [],
        },
    ),
    (
        "fidelity_gate_events",
        {
            "corpus": "forced",
            "row_pair_boundaries": 500,
            "counts": {"attempted": 0},
            "non_a": [],
        },
        {
            "corpus": "healthy",
            "row_pair_boundaries": 10,
            "counts": {"attempted": 10, "a": 10},
            "non_a": [],
        },
    ),
]

#: argv per harness. `--tables` is required by three of the four; the value is never read,
#: because `run_corpus` — the only consumer — is replaced.
ARGV = {
    "leaf_vs_reality": ["--corpus", "x", "--tables", "t"],
    "leaf_root_parity": ["--corpus", "x", "--tables", "t"],
    "prior_mapping_assert": ["--corpus", "x", "--tables", "t"],
    "fidelity_gate_events": ["--corpus", "x"],
}


def _run(module_name: str, report: dict) -> int:
    module = importlib.import_module(module_name)
    argv = list(ARGV[module_name])
    patches = [mock.patch.object(module, "run_corpus", return_value=report)]
    # Three of the four read a tables artifact before looping; stub the read, not the parse,
    # so any real argument validation still runs.
    if module_name != "fidelity_gate_events":
        patches.append(
            mock.patch.object(Path, "read_text", return_value='{"layout": {}, "vocab": {}}')
        )
    with patches[0]:
        if len(patches) > 1:
            with patches[1]:
                return module.main(argv)
        return module.main(argv)


class DenominatorAcceptanceTest(unittest.TestCase):
    def test_a_fully_skipped_corpus_exits_nonzero(self) -> None:
        """THE acceptance criterion, for all four harnesses."""
        for name, skipped, _healthy in HARNESSES:
            with self.subTest(harness=name):
                self.assertEqual(
                    _run(name, skipped),
                    1,
                    f"{name} reported a pass on a corpus where every boundary was skipped",
                )

    def test_each_harness_still_passes_when_it_measured_something(self) -> None:
        """Non-vacuity control. Without this, `return 1` unconditionally would satisfy the
        acceptance above while making every harness useless."""
        for name, _skipped, healthy in HARNESSES:
            with self.subTest(harness=name):
                self.assertEqual(
                    _run(name, healthy),
                    0,
                    f"{name} failed a clean run, so the acceptance test above proves nothing",
                )

    def test_an_unclassified_attempt_also_fails(self) -> None:
        """Rule 3, end to end: a boundary attempted and never classified must not pass.

        This is the half that a caller deriving `measured` from `matched + diverged` would
        lose, so it is pinned against the real harnesses rather than only the helper.
        """
        for name, _skipped, healthy in HARNESSES:
            with self.subTest(harness=name):
                broken = {**healthy, "counts": {**healthy["counts"], "attempted": 11}}
                self.assertEqual(
                    _run(name, broken),
                    1,
                    f"{name} passed with 11 attempts and only 10 classified",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
