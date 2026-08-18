"""Gates for the §8 acceptance merge/report path.

The acceptance run exists to settle a claim that a *pooled* number already hid
once (docs/mcts_degradation_findings.md §11), so the reporting path has three
jobs and each is pinned here:

* score the two seats SEPARATELY;
* refuse to score an incomplete mirrored pair (fail-closed, in-house rule from
  ``mcts_eval.scoring.pair_scores``) while still naming it;
* refuse to merge shards produced by two different engine builds.

Runs on shard fixtures — no cluster, no checkpoint, no torch.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "scripts" / "mcts_acceptance_report.py"

#: The reached-depth accumulation, as it appears on every leaf-eval search path.
_DEPTH_ACCUMULATION = r"depth_reached_histogram\[reached\] \+= "


def _depth_accumulation_sites(source: str, method: str) -> int:
    """How many reached-depth accumulations sit inside ``method``'s body.

    A FUNCTION OF ITS INPUTS, not a method reading the tree, so the guard that uses
    it can be shown to read False on a synthetic fourth search path. A checker that
    can only be pointed at the real file cannot demonstrate its own failing input,
    which is how the "a NEW leaf-eval path fails here" claim went unchecked.
    """

    marker = f"    def {method}("
    if marker not in source:
        return 0
    rest = source[source.index(marker) + 1 :]
    end = rest.find("\n    def ")
    body = rest if end < 0 else rest[:end]
    return len(re.findall(_DEPTH_ACCUMULATION, body))
FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def shard(
    path: Path,
    *,
    arm: str,
    config_id: str,
    pair_start: int,
    pairs: int,
    p1_outcome: str,
    p2_outcome: str,
    fingerprint: str = FINGERPRINT_A,
    drop: tuple[int, str] | None = None,
) -> Path:
    rows = []
    for index in range(pairs):
        seed = pair_start + index
        for seat, outcome in (("p1", p1_outcome), ("p2", p2_outcome)):
            if drop == (seed, seat):
                continue
            rows.append(
                {
                    "config_id": config_id,
                    "seed": seed,
                    "seat": seat,
                    "outcome": outcome,
                    "turns": 40,
                    "provenance_sha256": f"prov-{arm}",
                    "opponent_crashed": False,
                }
            )
    path.write_text(
        json.dumps(
            {
                "schema_version": "pokezero.mcts-acceptance-shard.v1",
                "arm": arm,
                "config_id": config_id,
                "checkpoint": "ckpt",
                "engine_fingerprint": fingerprint,
                "provenance_sha256": f"prov-{arm}",
                "pair_start": pair_start,
                "pairs": pairs,
                "games": len(rows),
                "total_decisions": 1,
                "fallback_decisions": 0,
                "fallback_rate": 0.0,
                "fallback_reasons": {},
                "world_failure_reasons": {},
                "search_wall_per_decision": 0.0,
                "wall_s": 1.0,
                "results": rows,
                "per_game": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def run_report(*paths: Path, extra: list[str] | None = None):
    return subprocess.run(
        [sys.executable, str(REPORT), *[str(p) for p in paths], *(extra or [])],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


class AcceptanceReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_seats_are_reported_separately(self) -> None:
        """A seat-split defect must be visible even when the pool looks fine.

        p1 wins every game and p2 loses every game: the pooled pair mean is
        exactly 0.500 — the same number a healthy arm would show — while the
        seats are 1.000 and 0.000. This is §11's failure mode in miniature.
        """
        shard(
            self.tmp / "s0.json",
            arm="search",
            config_id="d4-s1024-b64-w4",
            pair_start=7800000,
            pairs=20,
            p1_outcome="win",
            p2_outcome="loss",
        )
        result = run_report(self.tmp / "s0.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("p1 seat  n=  20  score=1.000", result.stdout)
        self.assertIn("p2 seat  n=  20  score=0.000", result.stdout)
        self.assertIn("pooled pair mean  0.500", result.stdout)

    def test_incomplete_pair_is_named_and_never_scored(self) -> None:
        shard(
            self.tmp / "s0.json",
            arm="search",
            config_id="d4-s1024-b64-w4",
            pair_start=7800000,
            pairs=10,
            p1_outcome="win",
            p2_outcome="win",
            drop=(7800004, "p2"),
        )
        result = run_report(self.tmp / "s0.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("complete pairs     : 9", result.stdout)
        self.assertIn("INCOMPLETE: [7800004]", result.stdout)
        # The surviving p1 game of the broken pair must not inflate the seat n.
        self.assertIn("p1 seat  n=   9", result.stdout)

    def test_shards_from_two_builds_are_refused(self) -> None:
        shard(
            self.tmp / "a.json",
            arm="search",
            config_id="d4-s1024-b64-w4",
            pair_start=7800000,
            pairs=5,
            p1_outcome="win",
            p2_outcome="loss",
        )
        shard(
            self.tmp / "b.json",
            arm="search",
            config_id="d4-s1024-b64-w4",
            pair_start=7800010,
            pairs=5,
            p1_outcome="win",
            p2_outcome="loss",
            fingerprint=FINGERPRINT_B,
        )
        result = run_report(self.tmp / "a.json", self.tmp / "b.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mixes 2 engine builds", result.stdout + result.stderr)

    def test_expected_fingerprint_is_enforced(self) -> None:
        shard(
            self.tmp / "a.json",
            arm="search",
            config_id="d4-s1024-b64-w4",
            pair_start=7800000,
            pairs=5,
            p1_outcome="win",
            p2_outcome="loss",
        )
        result = run_report(
            self.tmp / "a.json", extra=["--expect-fingerprint", FINGERPRINT_B]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("staged config expects", result.stdout + result.stderr)




class ModelPathDepthInstrumentationTest(unittest.TestCase):
    """The model path must accumulate reached depth, like the hp_fraction path.

    Without this a depth ladder cannot be interpreted: "depth does not help" and
    "the simulation budget never let the tree reach the cap" produce the same flat
    ladder, and only the reached-depth histogram separates them.
    """

    def test_model_path_accumulates_reached_depth(self) -> None:
        source = (REPO_ROOT / "src" / "pokezero" / "engine_search.py").read_text()
        # The two accumulation sites must both exist: the hp_fraction path and
        # the model path. Locate them by their neighbouring model-only counter.
        self.assertIn("self.stats.model_evals += int(report[\"model_evals\"])", source)
        model_block = source.split("self.stats.model_evals += int(report[\"model_evals\"])")[1][:900]
        for field in (
            "depth_reached_samples",
            "depth_reached_sum",
            "depth_reached_max",
            "depth_reached_histogram",
        ):
            self.assertIn(
                field,
                model_block,
                f"model path does not accumulate {field}; a depth ladder run on "
                "leaf_eval='model' would carry no reached-depth evidence",
            )

    def test_every_registered_leaf_eval_path_accumulates_reached_depth_once(self) -> None:
        """ONE SITE PER REGISTERED LEAF-EVAL PATH, derived rather than counted.

        The previous form read `== 2` ("hp_fraction and model"), the rollout seam
        legitimately added a third path, and the fix was to make the requirement per
        path -- with the path list written as a LITERAL TUPLE and the file-wide total
        compared against `len(` that same tuple. Both halves were then
        self-referential, and the stated claim -- "a NEW leaf-eval path fails here
        until it is instrumented" -- was FALSE: a fourth uninstrumented search
        method leaves the total at 3 and survived a 39-module sweep. That is
        demonstrated below, on this test's own checker.

        The expectation is now DERIVED from `LEAF_EVAL_SEARCH_METHODS`, and
        registration there is what makes a mode selectable at all
        (`EngineMctsConfig.__post_init__` validates against its keys). So a new
        leaf-eval path is either unregistered -- in which case no config can reach
        it and it is not a leaf-eval path -- or registered, in which case it appears
        here and must carry its own accumulation.
        """
        from pokezero import engine_search

        source = (REPO_ROOT / "src" / "pokezero" / "engine_search.py").read_text()
        methods = engine_search.DEPTH_INSTRUMENTED_SEARCH_METHODS
        self.assertGreaterEqual(
            len(methods),
            3,
            "the registry must still name the crate-backed leaf-eval paths; an "
            "empty derived set would make every assertion below vacuous",
        )
        for name in methods:
            with self.subTest(path=name):
                self.assertEqual(
                    _depth_accumulation_sites(source, name),
                    1,
                    f"{name} must accumulate the reached-depth histogram exactly "
                    "once; a leaf-eval path without it carries no depth evidence, "
                    "and one with two double-counts every world",
                )
        self.assertEqual(
            len(re.findall(_DEPTH_ACCUMULATION, source)),
            len(methods),
            "the file must hold exactly one accumulation per REGISTERED leaf-eval "
            f"search path ({', '.join(methods)}) and no others -- a site outside "
            "them is unaccounted for",
        )

    def test_the_registry_is_what_makes_a_leaf_eval_mode_selectable(self) -> None:
        """The half of the claim that lives outside this file.

        "A new leaf-eval path fails here until it is instrumented" is only true if a
        path that skips the registry cannot run. So: every registered key must
        construct, and an unregistered one must be refused by the config itself.
        """
        from pokezero.engine_search import EngineMctsConfig, LEAF_EVAL_SEARCH_METHODS

        for mode in LEAF_EVAL_SEARCH_METHODS:
            with self.subTest(mode=mode):
                # Some modes have further required knobs (`model` needs its
                # artifacts), so the assertion is on the MEMBERSHIP refusal
                # specifically -- a registered mode never trips it.
                try:
                    EngineMctsConfig(
                        leaf_eval=mode,
                        search_sims=1,
                        search_depth=1,
                        rollout_count=1,
                    )
                except ValueError as error:
                    self.assertNotIn("leaf_eval must be", str(error))
        with self.assertRaisesRegex(ValueError, "leaf_eval must be"):
            EngineMctsConfig(leaf_eval="fourth_pricer")

    def test_an_unregistered_path_is_the_hole_and_registering_it_closes_it(self) -> None:
        """THE DEMONSTRATED FAILING INPUT for this guard, in both directions.

        Applied to this test's own checker rather than to the tree, because the
        defect is a property of the checker: given a fourth search method with no
        accumulation, the OLD form (a literal method list plus a total compared
        against its length) reads True, and the derived form reads False the moment
        the method is registered.
        """
        source = (REPO_ROOT / "src" / "pokezero" / "engine_search.py").read_text()
        fourth = (
            "\n    def _search_fourth_pricer(self, context, worlds, rng):\n"
            '        """A leaf-eval path that reports no reached depth."""\n'
            "        return None\n"
        )
        mutant = source + fourth

        # The OLD shape: a hard-coded list of three, and a file-wide total compared
        # against len(that list). Both still hold -- which is the bug.
        legacy = ("_search_hp_fraction_crate", "_search_rollout_crate", "_search_model")
        self.assertEqual(len(re.findall(_DEPTH_ACCUMULATION, mutant)), len(legacy))
        for name in legacy:
            self.assertEqual(_depth_accumulation_sites(mutant, name), 1)

        # The DERIVED shape, with the fourth path registered: reads False.
        self.assertEqual(_depth_accumulation_sites(mutant, "_search_fourth_pricer"), 0)
        registered = legacy + ("_search_fourth_pricer",)
        self.assertNotEqual(
            len(re.findall(_DEPTH_ACCUMULATION, mutant)),
            len(registered),
            "with the fourth path registered the file-wide total must no longer "
            "match the registry -- otherwise the derived form is no stronger than "
            "the literal one it replaces",
        )

        # ... and instrumenting it makes the derived form pass again, so the check is
        # discriminating rather than merely refusing anything new.
        instrumented = source + fourth.replace(
            "        return None\n",
            "        self.stats.depth_reached_histogram[reached] += 1\n"
            "        return None\n",
        )
        self.assertEqual(
            _depth_accumulation_sites(instrumented, "_search_fourth_pricer"), 1
        )
        self.assertEqual(
            len(re.findall(_DEPTH_ACCUMULATION, instrumented)), len(registered)
        )

    def test_REGISTER_AND_STARVE_is_refused_rather_than_silently_substituted(
        self,
    ) -> None:
        """B5. A4's claim had a third horn, and it produced ZERO new failures.

        "A new leaf-eval path is either unregistered and unreachable, or registered
        and required to carry its own reached-depth accumulation." The `None` sentinel
        is a third state that satisfies neither: `DEPTH_INSTRUMENTED_SEARCH_METHODS`
        filters the `None`s out, so the instrumentation guard skips the mode;
        `__post_init__` accepts it, because membership is all it checks; and
        `_search`'s dispatch had no final branch, so control fell into the in-process
        `hp_fraction` tree. The mode was SELECTABLE, ran another path's leaf
        evaluator, and every artifact recorded its own name -- a row whose provenance
        field names a pricer that never ran, which is the false-witness class reached
        from the registry side.

        Both halves are asserted: the refusal fires on a registered-but-undispatched
        mode, and it does NOT fire on the one mode whose implementation legitimately
        IS the fall-through.
        """
        from pokezero.engine_search import (
            EngineMctsConfig,
            EngineSearchWitnessError,
            LEAF_EVAL_IN_PROCESS_MODE,
            LEAF_EVAL_SEARCH_METHODS,
            require_leaf_eval_dispatched,
        )

        # The fall-through's own mode passes, or the guard refuses production.
        require_leaf_eval_dispatched(LEAF_EVAL_IN_PROCESS_MODE)
        self.assertIn(LEAF_EVAL_IN_PROCESS_MODE, LEAF_EVAL_SEARCH_METHODS)
        self.assertIsNone(LEAF_EVAL_SEARCH_METHODS[LEAF_EVAL_IN_PROCESS_MODE])

        # THE ATTACK: registered with the `None` sentinel, hence exempt from the
        # instrumentation guard, hence starved of a dispatch branch.
        starved = "fourth_pricer"
        self.assertNotIn(starved, LEAF_EVAL_SEARCH_METHODS)
        with self.assertRaises(EngineSearchWitnessError) as caught:
            require_leaf_eval_dispatched(starved)
        message = str(caught.exception)
        self.assertIn(starved, message)
        self.assertIn(LEAF_EVAL_IN_PROCESS_MODE, message)
        self.assertIn("does not give it a search", message)

        # And EVERY registered mode is either dispatched by a branch in `_search` or
        # is the fall-through -- derived from the registry, so adding an entry to it is
        # what makes this read False.
        source = (REPO_ROOT / "src" / "pokezero" / "engine_search.py").read_text()
        dispatch = source[
            source.index("        self.stats.world_search_attempts += len(worlds)") :
            source.index("        require_leaf_eval_dispatched(")
        ]
        for mode in LEAF_EVAL_SEARCH_METHODS:
            with self.subTest(mode=mode):
                if mode == LEAF_EVAL_IN_PROCESS_MODE:
                    continue
                self.assertIn(
                    f'self._config.leaf_eval == "{mode}"',
                    dispatch,
                    f"{mode!r} is registered and selectable but has no dispatch "
                    "branch, so it would be priced by another path's leaf evaluator",
                )
        # The registry-derived form reads False on the synthetic fourth mode, which is
        # what makes it stronger than an enumeration.
        self.assertNotIn(f'self._config.leaf_eval == "{starved}"', dispatch)
        with self.assertRaisesRegex(ValueError, "leaf_eval must be"):
            EngineMctsConfig(leaf_eval=starved)

    def test_runner_emits_the_policy_stats_payload(self) -> None:
        source = (REPO_ROOT / "scripts" / "mcts_acceptance_h2h.py").read_text()
        self.assertIn("policy_stats", source)
        self.assertIn("to_payload()", source)


if __name__ == "__main__":  # pragma: no cover
    # At the END. It sat at line 185, stranding ModelPathDepthInstrumentationTest
    # from direct execution -- found by the repo-wide structural guard in
    # tests/test_public_invariant.py.
    unittest.main()
