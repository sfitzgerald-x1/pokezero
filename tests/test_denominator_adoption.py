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
            # `skip:forced` present because the real run_corpus cannot emit attempted == 0
            # WITHOUT skips -- an earlier revision's fixture had `row_pair_boundaries: 500`
            # with a bare `attempted: 0`, an arithmetically impossible pair, so the subtest
            # was green on both the correct and the broken harness. See the real-driver test
            # below, which is the layer that actually caught it.
            "counts": {"attempted": 0, "skip:forced": 500},
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

        Note what this is and is not: after the round-1 fixes all four increments sit
        adjacent to their classification, so no harness can actually PRODUCE this state. It
        drives real `main()`s, but it is helper arithmetic in a harness costume — a guard
        against a future refactor, not evidence about today's accounting. Rule 4
        (`measured + skipped == contained`) is the one that bites, and it is pinned in
        tests/test_differential_denominator.py against both round-1 bugs.
        """
        for name, _skipped, healthy in HARNESSES:
            with self.subTest(harness=name):
                broken = {**healthy, "counts": {**healthy["counts"], "attempted": 11}}
                self.assertEqual(
                    _run(name, broken),
                    1,
                    f"{name} passed with 11 attempts and only 10 classified",
                )


class GuardsArePinnedTest(unittest.TestCase):
    """The guards added for rule 4 -- without this, all three revert green.

    Review observed that the commit adding them touched no test file: deleting the
    vocabulary `raise`, deleting the `NOT CHECKED` phrase, or reverting `contained=report[...]`
    to `.get` each left every test passing. That is the "guard nothing pins" pattern that
    produced the original blocker in this PR, so it is closed here rather than noted.
    """

    def _denominator_lines(self, name: str, report: dict) -> list[str]:
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _run(name, report)
        return [l for l in buffer.getvalue().splitlines() if "[denominator]" in l]

    def test_every_adoption_site_keeps_rule_4_armed(self) -> None:
        """Pins that all four supply BOTH `contained` and `skipped`.

        Reverting any site to `report.get(...)` and renaming the key, or dropping `skipped=`,
        silently disarms rule 4 -- the only load-bearing rule -- and this goes red.
        """
        for name, _skipped, healthy in HARNESSES:
            with self.subTest(harness=name):
                lines = self._denominator_lines(name, healthy)
                self.assertTrue(lines, f"{name} printed no denominator line at all")
                line = lines[0]
                # assertRegex, not assertIn: review showed the plain substrings are
                # UNFALSIFIABLE here, because the disarmed line reads "rule 4 NOT CHECKED
                # (contained/skipped not supplied)" and contains both words. Both assertIns
                # passed on a disarmed line. Match the armed forms -- "of N contained",
                # "N skipped" -- which only render() with both figures present can produce.
                self.assertRegex(line, r"of \d+ contained", f"{name} does not supply `contained`")
                self.assertRegex(line, r"\d+ skipped", f"{name} does not supply `skipped`")
                self.assertNotIn(
                    "NOT CHECKED", line, f"{name} is running with rule 4 disarmed"
                )

    def test_a_disarmed_run_says_so_in_its_output(self) -> None:
        """Non-vacuity for the assertion above: the phrase it looks for really does appear
        when rule 4 is off, so `assertNotIn` is discriminating rather than always-true."""
        import io
        from contextlib import redirect_stdout

        from differential_denominator import check_denominator, gate

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            gate([check_denominator("bare", measured=5, matched=2, diverged=3)])
        self.assertIn("NOT CHECKED", buffer.getvalue())

    def test_the_vocabulary_guard_is_pinned_without_a_corpus(self) -> None:
        """The same guard as the test below, but reachable in CI.

        Review found the guard's only pin was corpus-gated and the corpus is gitignored: with
        the `raise` deleted, the CI step still read `Ran 28 / OK (skipped=6)`. Two of the three
        guards this class exists to pin were covered in CI and this one was not. It is now at
        module scope so it can be called directly.
        """
        module = importlib.import_module("fidelity_gate_events")
        # Clean vocabularies pass, including a skip reason nobody has written yet.
        module.assert_status_vocabulary({"a": 5, "b": 1, "c": 2, "attempted": 8, "skip:novel": 3})
        for bad in ("unsupported:new_reason", "d", "skipped:typo", "skip_typo"):
            with self.subTest(status=bad):
                with self.assertRaises(AssertionError) as caught:
                    module.assert_status_vocabulary({"a": 1, "attempted": 1, bad: 1})
                # f"['{bad}']", not `bad`: the bare letter "d" occurs in "returned",
                # "verdict" and "denominator", so assertIn(bad) is non-discriminating for
                # that case. Match the rendered `sorted(unknown)` list instead.
                self.assertIn(f"['{bad}']", str(caught.exception))

    def test_run_corpus_actually_calls_the_vocabulary_guard(self) -> None:
        """Hoisting the guard to module scope made a NEW failure mode expressible.

        While it was inline, "not called" could not happen. Now deleting the call while
        leaving the function intact is a live mutation, and review showed it is CI-invisible:
        corpus-present FAILED, corpus-absent `Ran 30 / OK (skipped=6)` with both greps
        matching. So the call site gets its own corpus-free pin.

        This is a SOURCE-SHAPE assertion, not a behavioural one -- it cannot tell whether the
        call runs on a given input. The corpus-driven test below is the behavioural proof; this
        one exists so CI is not blind to the deletion.

        It checks POSITION as well as presence, because "unconditional and before every return"
        is not enough. Review moved the call to just above the loop, where it is top-level,
        unconditional and pre-return but inspects an empty Counter -- a permanent no-op that
        passed. That is the SAME defect run_corpus already shipped once and that the comment
        four lines above the call exists to explain: a check placed above the loop that produces
        the values it checks. So the call must sit AFTER the loop, and must be passed `stats`.

        THIS IS A CHANGE-DETECTOR, deliberately. Review confirmed four behaviour-preserving
        refactors it rejects as false positives -- `dict(stats)`, `stats2 = stats`, `*[stats]`,
        and appending any unrelated `for` loop below the call (which makes the lineno check
        harder, not easier, and reports a misleading reason). The trade is accepted because the
        thing it protects has been silently disarmed twice in this PR, and a false positive
        costs one confused minute and a one-line edit. If you hit one, that is this comment, not
        a bug.

        KNOWN SURVIVOR, not chased: rebinding `stats = Counter()` on the line above the call
        passes all five assertions while the guard inspects an empty counter. Closing it means
        reasoning about dataflow, i.e. writing a static analyser inside a unit test. Unlike the
        hoist above, it is not a line anyone writes by accident, and the corpus-driven tests
        catch it twice. The honest boundary is source shape, not behaviour.
        """
        import ast
        import inspect
        import textwrap

        module = importlib.import_module("fidelity_gate_events")
        tree = ast.parse(textwrap.dedent(inspect.getsource(module.run_corpus)))
        func = tree.body[0]
        calls = [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assert_status_vocabulary"
        ]
        self.assertEqual(len(calls), 1, "run_corpus must call the vocabulary guard exactly once")
        # Unconditional and before every return, so no path can reach a result past it.
        self.assertTrue(
            any(
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and getattr(stmt.value.func, "id", None) == "assert_status_vocabulary"
                for stmt in func.body
            ),
            "the guard call must sit at run_corpus's top level, not inside a branch",
        )
        returns = [n.lineno for n in ast.walk(func) if isinstance(n, ast.Return)]
        self.assertTrue(returns, "run_corpus has no return -- this test is looking at the wrong function")
        self.assertLess(
            calls[0].lineno, min(returns), "the guard must run before run_corpus can return"
        )
        # It must inspect `stats` itself, not a literal or some other name.
        self.assertEqual(
            getattr(calls[0].args[0], "id", None) if calls[0].args else None,
            "stats",
            "the guard must be passed `stats`, not a stand-in that can never contain a status",
        )
        # And it must run AFTER the loop that fills `stats`. Above it, the Counter is empty and
        # the guard is a permanent no-op -- the round-1 defect, repeated one level up.
        loops = [n.lineno for n in ast.walk(func) if isinstance(n, ast.For)]
        self.assertTrue(loops, "run_corpus no longer has the loop that fills `stats`")
        self.assertGreater(
            calls[0].lineno,
            max(loops),
            "the guard sits above the loop that fills `stats`, so it inspects an empty Counter",
        )

    def test_an_unprefixed_status_raises_rather_than_inflating_measured(self) -> None:
        """The vocabulary guard, through the REAL run_corpus.

        Rule 4 cannot catch this: moving a boundary from `skipped` to `measured` changes both
        sides of `measured + skipped == contained` by one each, so they cancel. Measured on
        golden-v4 before the guard: 1064 + 207 = 1271, all four rules green, 424 boundaries
        never driven. Hence a guard, and hence this pin.
        """
        corpus = REPO_ROOT / "corpus" / "golden-v4"
        if not (corpus / "rows.jsonl").exists():
            self.skipTest(f"no corpus at {corpus}")
        module = importlib.import_module("fidelity_gate_events")
        real = module.drive_boundary
        calls = {"n": 0}

        def sometimes_unprefixed(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] % 3 == 0:
                return module.BoundaryResult("unsupported:new_reason", "forced")
            return real(*args, **kwargs)

        with mock.patch.object(module, "drive_boundary", sometimes_unprefixed):
            with self.assertRaises(AssertionError) as caught:
                module.main(["--corpus", str(corpus)])
        self.assertIn("unsupported:new_reason", str(caught.exception))


class RealDriverAcceptanceTest(unittest.TestCase):
    """The same acceptance, one seam LOWER — against the real `run_corpus`.

    The class above patches `run_corpus`, which pins `main()`'s wiring but never the
    `attempted` counter that feeds it. That gap hid a real bug: `fidelity_gate_events`
    incremented `attempted` BEFORE its skip decision (which lives inside `drive_boundary`),
    making `measured` identically the row-pair count — so a 100%-skipped run still exited 0
    while the fixture-level subtest passed. Independent review found it by simulating one
    level down, which is what this class now does.

    Requires the committed v4 corpus; skips without it rather than silently proving nothing.
    """

    CORPUS = REPO_ROOT / "corpus" / "golden-v4"
    TABLES = REPO_ROOT / "corpus" / "encoder_tables_v4.json"

    def setUp(self) -> None:
        if not (self.CORPUS / "rows.jsonl").exists() or not self.TABLES.exists():
            self.skipTest(f"no corpus at {self.CORPUS}; regenerate per C112's provenance block")

    def test_leaf_root_parity_fails_when_every_boundary_skips(self) -> None:
        """Forced by making the encoder unconstructible, so every row lands in a skip."""
        module = importlib.import_module("leaf_root_parity")
        import pokezero_search

        def boom(*_a, **_k):
            raise RuntimeError("forced by the acceptance test")

        with mock.patch.object(pokezero_search, "LeafEncoder", boom):
            self.assertEqual(
                module.main(["--corpus", str(self.CORPUS), "--tables", str(self.TABLES)]), 1
            )

    def test_prior_mapping_assert_fails_when_every_boundary_skips(self) -> None:
        module = importlib.import_module("prior_mapping_assert")
        import pokezero_search

        def boom(*_a, **_k):
            raise RuntimeError("forced by the acceptance test")

        with mock.patch.object(pokezero_search, "LeafEncoder", boom):
            self.assertEqual(
                module.main(["--corpus", str(self.CORPUS), "--tables", str(self.TABLES)]), 1
            )

    def test_leaf_vs_reality_fails_on_a_genuine_schema_mismatch(self) -> None:
        """No patch at all: a v3 corpus against v4 tables skips 1008/1008 for real."""
        module = importlib.import_module("leaf_vs_reality")
        v3_corpus = REPO_ROOT / "corpus" / "golden-v2"
        if not (v3_corpus / "rows.jsonl").exists():
            self.skipTest("no corpus/golden-v2")
        self.assertEqual(
            module.main(["--corpus", str(v3_corpus), "--tables", str(self.TABLES)]), 1
        )

    def test_fidelity_gate_events_fails_when_every_boundary_skips(self) -> None:
        module = importlib.import_module("fidelity_gate_events")

        class _Skipped:
            status = "skip:forced"
            detail = "forced by the acceptance test"

        with mock.patch.object(module, "drive_boundary", return_value=_Skipped()):
            self.assertEqual(
                module.main(["--corpus", str(self.CORPUS)]),
                1,
                "the real run_corpus produced a 100%-skipped report and the harness passed",
            )

    def test_and_the_same_corpus_passes_the_denominator_undriven(self) -> None:
        """Non-vacuity: without the patch this corpus measures 956 of the 1271 boundaries it
        contains (315 skipped) and the denominator is clean, so the failure above is the
        forced skip and not the corpus."""
        module = importlib.import_module("fidelity_gate_events")
        self.assertEqual(module.main(["--corpus", str(self.CORPUS)]), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
