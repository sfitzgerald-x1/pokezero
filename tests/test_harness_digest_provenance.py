"""Pins for the harness digest: the Python half of the measuring instrument.

WHAT THIS MODULE IS FOR. `scripts/engine_build_fingerprint.py` identifies the
native engine. It covers the patch set, `BASE_SOURCE` and the search crate's
sources, and it covers NOTHING under `scripts/` or `src/pokezero/` -- correctly,
because that set is a REBUILD TRIGGER and adding Python to it would fire the
60-minute `mass-gate` engine build on every prose edit. The consequence is that a
sweep number's provenance names the engine and not the harness that produced it,
and the owner's deferred-sweep precondition ("the ledger is terminal and the
engine fingerprint is frozen", `RATIFIED_SWEEP_PRECONDITION`) therefore cannot
bear the weight placed on it: freezing the fingerprint does not freeze the
instrument.

`scripts/harness_digest.py` closes that. This module pins it, and pins the drift
it exists to make assertable.

RUNS WITHOUT THE ENGINE, deliberately, and that constrains how it is written. Its
CI job (`harness-provenance` in `.github/workflows/engine-fidelity-gates.yml`) is a
bare checkout plus `setup-python` -- no build, no install -- so
`import engine_transition_differential` would raise `ModuleNotFoundError` here
while passing on any dev machine. The differential is therefore inspected with
`ast`, never imported. `scripts/harness_digest.py` and
`scripts/engine_build_fingerprint.py` ARE imported, by file location: both are
stdlib-only at import time.
"""

from __future__ import annotations

import ast
import glob
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIFFERENTIAL = REPO / "scripts" / "engine_transition_differential.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


harness = _load("harness_digest_under_test", REPO / "scripts" / "harness_digest.py")
fingerprint = _load(
    "engine_build_fingerprint_under_test", REPO / "scripts" / "engine_build_fingerprint.py"
)


# ---------------------------------------------------------------------------
# The closure.
# ---------------------------------------------------------------------------

# MEASURED by running `harness_files()` itself, not transcribed from a reading of
# the import statements. EXACT rather than a floor, and the reason is the one the
# sibling corpus pins record: a member VANISHING is this shape's fail-open, and a
# floor -- or a pure-addition check -- masks it. If the differential stops
# importing the matcher because the matcher moved, the closure shrinks, the digest
# silently stops covering it, and every downstream claim about what the digest
# covers becomes false while staying green.
#
# Both directions are failures worth a human. Growth means the instrument now
# reaches code nobody declared part of it; shrinkage means it stopped reaching code
# that was. Bumping this set is only correct after confirming the set DIFFERENCE
# against the base tree is exactly what the change intends.
#
# `scripts/harness_digest.py` is in its own closure because the differential
# imports it. That is deliberate: tampering with the digest moves the digest.
#
# 16 -> 73, AND THIS PIN COULD NOT HAVE CAUGHT THE REASON. Review found that the
# resolver dropped every relative import, so the closure was 16 files where the
# honest answer is 73. A membership pin listing the 16 that ARE found stays green
# forever over a truncated graph -- it can only detect growth that never happened,
# which is precisely the fail-open the comment above claims it closes. The comment
# is kept because it is true for the case it describes (a member LEAVING), and the
# case it cannot describe is now covered by `ClosureCompletenessTests` below, which
# re-derives relative-import targets with an independent implementation.
#
# The count agrees with runtime: importing the differential in a built venv loads
# 72 first-party modules, all 73 - 1 of these, the extra being `inference_service`
# (imported lazily, so not resident at import time). Static superset of runtime,
# which is the safe direction.
_EXPECTED_HARNESS_CLOSURE = frozenset(
    {
        "scripts/differential_denominator.py",
        "scripts/engine_build_fingerprint.py",
        "scripts/engine_transition_differential.py",
        "scripts/fidelity_gate_events.py",
        "scripts/harness_digest.py",
        "src/pokezero/__init__.py",
        "src/pokezero/actions.py",
        "src/pokezero/admission_guard.py",
        "src/pokezero/audit_provenance.py",
        "src/pokezero/belief.py",
        "src/pokezero/bootstrap.py",
        "src/pokezero/category_vocab.py",
        "src/pokezero/collection.py",
        "src/pokezero/dataset.py",
        "src/pokezero/determinization.py",
        "src/pokezero/dex.py",
        "src/pokezero/diversity_population.py",
        "src/pokezero/engine_fidelity.py",
        "src/pokezero/engine_fidelity_multiturn.py",
        "src/pokezero/engine_search.py",
        "src/pokezero/engine_world.py",
        "src/pokezero/env.py",
        "src/pokezero/evaluation.py",
        "src/pokezero/evaluation_profiles.py",
        "src/pokezero/gen3_damage.py",
        "src/pokezero/golden_corpus.py",
        "src/pokezero/golden_corpus_fold.py",
        "src/pokezero/inference_service.py",
        "src/pokezero/investment.py",
        "src/pokezero/linear_policy.py",
        "src/pokezero/local_showdown.py",
        "src/pokezero/mcts_diagnostics.py",
        "src/pokezero/neural_policy.py",
        "src/pokezero/neural_selfplay.py",
        "src/pokezero/observation.py",
        "src/pokezero/opponents.py",
        "src/pokezero/padding.py",
        "src/pokezero/paths.py",
        "src/pokezero/poke_engine_adapter.py",
        "src/pokezero/poke_engine_backend.py",
        "src/pokezero/policy.py",
        "src/pokezero/promotion.py",
        "src/pokezero/public_action_capture.py",
        "src/pokezero/public_decision_corpus.py",
        "src/pokezero/randbat.py",
        "src/pokezero/randbat_vocab.py",
        "src/pokezero/refutation_curriculum.py",
        "src/pokezero/refutation_mining.py",
        "src/pokezero/refutation_population.py",
        "src/pokezero/refutation_progress.py",
        "src/pokezero/refutation_training.py",
        "src/pokezero/replay_benchmark.py",
        "src/pokezero/replay_branching.py",
        "src/pokezero/replay_import.py",
        "src/pokezero/rollout.py",
        "src/pokezero/root_puct_telemetry.py",
        "src/pokezero/run_audit.py",
        "src/pokezero/run_manifest.py",
        "src/pokezero/search.py",
        "src/pokezero/search_benchmark.py",
        "src/pokezero/search_policy.py",
        "src/pokezero/selfplay.py",
        "src/pokezero/shaping.py",
        "src/pokezero/showdown.py",
        "src/pokezero/showdown_fixture.py",
        "src/pokezero/source_metadata.py",
        "src/pokezero/teacher_scenarios.py",
        "src/pokezero/tier2.py",
        "src/pokezero/trajectory.py",
        "src/pokezero/transitions.py",
        "src/pokezero/transitions_fold.py",
        "src/pokezero/turn_merged.py",
        "src/pokezero/value_calibration.py",
    }
)


class TheClosureIsWhatWeThinkItIsTests(unittest.TestCase):
    def test_the_harness_closure_is_exactly_the_expected_set(self) -> None:
        found = {p.relative_to(REPO).as_posix() for p in harness.harness_files()}
        self.assertEqual(
            found,
            set(_EXPECTED_HARNESS_CLOSURE),
            "the harness import closure changed. Growth means the measuring "
            "instrument now reaches undeclared code; SHRINKAGE means the digest "
            "silently stopped covering something it claims to cover, which is this "
            "pin's fail-open. Update _EXPECTED_HARNESS_CLOSURE only after confirming "
            "the set difference is exactly what the change intends.",
        )

    def test_the_closure_covers_the_three_named_halves_of_the_instrument(self) -> None:
        """Anti-vacuity. A set pin passes for any set, including a wrong one.

        World model, matcher and Showdown adapter are the three things the gap was
        stated in terms of. Named individually so that a future closure edit that
        keeps the count while dropping one of them is red.
        """

        found = {p.relative_to(REPO).as_posix() for p in harness.harness_files()}
        for required in (
            "src/pokezero/engine_world.py",
            "src/pokezero/engine_fidelity.py",
            "src/pokezero/engine_fidelity_multiturn.py",
            "src/pokezero/local_showdown.py",
            "src/pokezero/poke_engine_adapter.py",
        ):
            with self.subTest(required=required):
                self.assertIn(required, found)

    def test_every_member_of_the_closure_is_a_tracked_file(self) -> None:
        for path in harness.harness_files():
            self.assertTrue(path.is_file(), path)


def _relative_targets_naively(path: Path, base: Path = REPO) -> set[str]:
    """Every first-party file a `from . import` in `path` can denote.

    DELIBERATELY AN INDEPENDENT, DUMBER IMPLEMENTATION than
    `harness_digest._absolutize`. If this reused the resolver under test, a class
    of import the resolver drops would be dropped here too and the completeness
    assertion would agree with the bug -- which is exactly how the 16-file closure
    passed its own membership pin. This one walks up `level` directories from the
    file's own parent and looks for a matching `.py` or package directory on disk,
    with no notion of dotted package names at all.
    """

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        # NOT named `base`: that is the relativization root parameter, and shadowing
        # it here silently reported every target relative to the wrong directory.
        # Caught by the nested-subpackage assertion added below, which is the first
        # thing that ever passed a non-default `base`.
        origin = path.parent
        for _ in range(node.level - 1):
            origin = origin.parent
        heads = [origin / part for part in ((node.module or "").split(".") if node.module else [])]
        anchor = heads[-1] if heads else origin
        candidates = [anchor.with_suffix(".py"), anchor / "__init__.py"]
        candidates += [anchor / f"{a.name}.py" for a in node.names]
        candidates += [anchor / a.name / "__init__.py" for a in node.names]
        for candidate in candidates:
            if candidate.is_file():
                try:
                    found.add(candidate.resolve().relative_to(base.resolve()).as_posix())
                except ValueError:
                    continue
    return found


class ClosureCompletenessTests(unittest.TestCase):
    """THE PIN THAT WOULD HAVE CAUGHT THE TRUNCATION, and the reason the membership
    pin alone is not enough.

    A closure must be CLOSED: every first-party file reachable from a member must
    itself be a member. The membership pin cannot express that -- it lists what the
    resolver found, so a resolver that finds too little agrees with it perfectly.
    This asserts the invariant directly, using a resolver that shares no code with
    the one under test.
    """

    def test_the_closure_is_closed_under_relative_imports(self) -> None:
        members = {p.resolve().relative_to(REPO).as_posix() for p in harness.harness_files()}
        missing: dict[str, set[str]] = {}
        for member in sorted(members):
            targets = _relative_targets_naively(REPO / member)
            gap = targets - members
            if gap:
                missing[member] = gap
        self.assertEqual(
            missing,
            {},
            "the harness closure is NOT closed: these members relatively-import "
            "first-party files that the digest does not hash. That is the exact "
            "defect review found -- a resolver that drops an import class truncates "
            "the closure while every membership pin stays green.",
        )

    def test_the_completeness_check_is_not_vacuous(self) -> None:
        """It must actually find relative imports, or it asserts nothing.

        `src/pokezero/**` carries 70 relative `ImportFrom` statements; a version of
        this scan that found zero would pass the pin above over any closure at all.
        """

        total = sum(
            len(_relative_targets_naively(p)) for p in harness.harness_files()
        )
        self.assertGreater(total, 50, "the naive relative-import scan found almost nothing")

    def test_the_resolver_actually_follows_a_relative_import(self) -> None:
        """A direct unit test on a synthetic package, independent of this repo.

        The truncation survived because nothing tested the resolver's relative arm
        in isolation; it was only ever exercised through a closure whose expected
        value had been copied from the buggy run.
        """

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            pkg = root / "src" / "pokezero"
            pkg.mkdir(parents=True)
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "leaf.py").write_text("VALUE = 1\n", encoding="utf-8")
            (pkg / "mid.py").write_text(
                "from .leaf import VALUE\nfrom .deep.nested import DEEP\n", encoding="utf-8"
            )
            # A SUBPACKAGE, with an `..` import back out of it. No member of the real
            # closure lives below `src/pokezero/<mod>.py` today -- `mcts_eval` and
            # `scenario_studio` exist but the differential reaches neither -- so the
            # nested arm of BOTH the resolver and the naive completeness scan is
            # otherwise never exercised. Untested code that only runs after someone
            # else's unrelated change is how the first truncation shipped.
            deep = pkg / "deep"
            deep.mkdir()
            (deep / "__init__.py").write_text("", encoding="utf-8")
            (deep / "sibling.py").write_text("S = 1\n", encoding="utf-8")
            # Reachable ONLY through the `..` ascent, so the ascend arm is
            # load-bearing. An earlier version of this fixture pointed `..` at
            # `leaf.py`, which `mid.py` already imports with a single dot -- so
            # deleting the resolver's ascend loop changed no closure member and the
            # mutation came back green. The assertion existed and proved nothing.
            (pkg / "only_via_parent.py").write_text("Z = 1\n", encoding="utf-8")
            (deep / "nested.py").write_text(
                "from ..leaf import VALUE\n"
                "from ..only_via_parent import Z\n"
                "from .sibling import S\n"
                "DEEP = 1\n",
                encoding="utf-8",
            )
            (root / "scripts" / "root.py").write_text(
                "from pokezero.mid import VALUE\n", encoding="utf-8"
            )
            shutil.copy(REPO / "scripts" / "harness_digest.py", root / "scripts" / "harness_digest.py")
            module = _load("harness_digest_synthetic", root / "scripts" / "harness_digest.py")
            found = {
                p.resolve().relative_to(root.resolve()).as_posix()
                for p in module.harness_files(root / "scripts" / "root.py")
            }
            self.assertIn(
                "src/pokezero/leaf.py",
                found,
                "the resolver did not follow `from .leaf import VALUE` -- this is the "
                "truncation review caught, reproduced in five files",
            )
            self.assertIn("src/pokezero/mid.py", found)
            self.assertIn(
                "src/pokezero/deep/nested.py", found, "did not descend into a subpackage"
            )
            self.assertIn(
                "src/pokezero/deep/sibling.py",
                found,
                "did not follow `from .sibling import S` INSIDE a subpackage",
            )
            self.assertIn(
                "src/pokezero/only_via_parent.py",
                found,
                "did not follow `from ..only_via_parent import Z` -- the ascend arm "
                "is unexercised by the real closure, which has no subpackage members",
            )

            # The naive completeness scan must agree on the nested case too, or the
            # pin that guards the resolver carries the resolver's own blind spot.
            nested = _relative_targets_naively(deep / "nested.py", base=root)
            self.assertIn(
                "src/pokezero/leaf.py",
                nested,
                "the independent scan did not resolve `..leaf` out of a subpackage",
            )
            self.assertIn("src/pokezero/deep/sibling.py", nested)
            self.assertIn("src/pokezero/only_via_parent.py", nested)


# ---------------------------------------------------------------------------
# The digest itself. Sensitivity in BOTH directions, proved by construction on a
# copy of the tree rather than asserted.
# ---------------------------------------------------------------------------


class _TreeCopy:
    """A throwaway copy of the two source roots, with a working harness resolver.

    `harness_digest` resolves imports against its own `REPO_ROOT`, so the module is
    re-imported from inside the copy. That is what makes these tests exercise the
    real resolver rather than a reimplementation of it.
    """

    def __init__(self) -> None:
        self.tmp = tempfile.mkdtemp()
        root = Path(self.tmp)
        shutil.copytree(REPO / "scripts", root / "scripts")
        (root / "src").mkdir()
        shutil.copytree(REPO / "src" / "pokezero", root / "src" / "pokezero")
        self.root = root
        self._n = 0

    def digest(self) -> str:
        self._n += 1
        module = _load(
            f"harness_digest_copy_{self._n}", self.root / "scripts" / "harness_digest.py"
        )
        return str(module.compute_harness_digest()["harness_digest"])

    def close(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class TheDigestActuallyMeasuresSomethingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = _TreeCopy()
        self.addCleanup(self.tree.close)

    def test_the_digest_is_deterministic(self) -> None:
        self.assertEqual(self.tree.digest(), self.tree.digest())

    def test_a_copy_of_the_tree_reproduces_the_live_digest(self) -> None:
        """Location independence, which is what makes the digest comparable at all."""

        self.assertEqual(self.tree.digest(), harness.harness_digest())

    def test_the_digest_moves_when_the_matcher_changes(self) -> None:
        """THE POINT OF THE WHOLE MODULE, proved rather than asserted.

        This is the exact drift that was confirmed live: a counter added to the
        differential after C142's sweeps changed what a sweep reports while the
        engine fingerprint stayed `5fa147ff`.
        """

        before = self.tree.digest()
        matcher = self.tree.root / "src" / "pokezero" / "engine_fidelity.py"
        matcher.write_text(
            matcher.read_text(encoding="utf-8") + "\n# harness drift probe\n",
            encoding="utf-8",
        )
        self.assertNotEqual(before, self.tree.digest())

    def test_the_digest_moves_when_the_differential_changes(self) -> None:
        before = self.tree.digest()
        target = self.tree.root / "scripts" / "engine_transition_differential.py"
        target.write_text(
            target.read_text(encoding="utf-8") + "\n# harness drift probe\n", encoding="utf-8"
        )
        self.assertNotEqual(before, self.tree.digest())

    def test_the_digest_moves_when_the_digest_module_itself_changes(self) -> None:
        """Self-coverage: tampering with the instrument's identity moves it."""

        before = self.tree.digest()
        target = self.tree.root / "scripts" / "harness_digest.py"
        target.write_text(
            target.read_text(encoding="utf-8") + "\n# harness drift probe\n", encoding="utf-8"
        )
        self.assertNotEqual(before, self.tree.digest())

    def test_the_digest_does_not_move_for_a_script_outside_the_closure(self) -> None:
        """The other direction, and the reason this is a closure and not a glob.

        A digest that moved on every `scripts/` edit would be a rebuild trigger by
        another name, and the pressure to widen it back out is exactly how the
        engine fingerprint's own over-capture note got written.
        """

        before = self.tree.digest()
        (self.tree.root / "scripts" / "not_in_the_harness.py").write_text(
            "# not imported by the differential\n", encoding="utf-8"
        )
        unrelated = self.tree.root / "src" / "pokezero" / "fleet_worker.py"
        self.assertTrue(unrelated.is_file(), "fixture assumption: fleet_worker.py exists")
        self.assertNotIn(
            "src/pokezero/fleet_worker.py",
            _EXPECTED_HARNESS_CLOSURE,
            "fixture assumption: fleet_worker.py is outside the closure",
        )
        unrelated.write_text(
            unrelated.read_text(encoding="utf-8") + "\n# unrelated edit\n", encoding="utf-8"
        )
        self.assertEqual(before, self.tree.digest())


# ---------------------------------------------------------------------------
# The wiring. Inspected with `ast` because this job has no engine to import.
# ---------------------------------------------------------------------------


def _differential_tree() -> ast.Module:
    return ast.parse(DIFFERENTIAL.read_text(encoding="utf-8"))


def _function(name: str) -> ast.FunctionDef:
    for node in _differential_tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not a top-level function of {DIFFERENTIAL.name}")


def _returned_dict_keys(func: ast.FunctionDef) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
    return keys


def _module_frozenset(name: str) -> set[str]:
    for node in _differential_tree().body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if name not in targets:
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id == "frozenset" and value.args:
                literal = value.args[0]
                if isinstance(literal, (ast.Set, ast.List, ast.Tuple)):
                    return {
                        e.value
                        for e in literal.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    }
    raise AssertionError(f"{name} is not a module-level frozenset literal")


class TheDifferentialStampsItTests(unittest.TestCase):
    def test_checkpoint_provenance_records_the_harness_digest(self) -> None:
        """Without this the digest exists and nothing writes it -- an inert pin.

        This repo has found four of those, the last a CI job whose result nothing
        consumed. The assertion is on the producer, because no committed artifact
        carries the field yet and a pin over the corpus alone would be vacuous.
        """

        keys = _returned_dict_keys(_function("_checkpoint_provenance"))
        self.assertIn("harness_digest", keys)
        # The negative control: excluding one key must not have excluded the rest.
        for existing in ("source_commit", "engine_fingerprint", "source_tree", "enumerate_rolls"):
            with self.subTest(existing=existing):
                self.assertIn(existing, keys)

    def test_the_harness_digest_gates_a_resume(self) -> None:
        """It must NOT be demoted to descriptive the way `source_tree` was.

        `source_tree` is excluded from resume identity because it flips on any
        tracked edit and made a 200-game crash-safe sweep unresumable after a README
        touch. The harness digest covers the 73 files the differential imports and
        nothing else, so it does not move on prose, and a change to one of them
        mid-sweep is exactly the corruption a resume must refuse rather than merge
        into a report describing neither instrument. The churn that the wider
        closure does add was measured, not waved away: 7 of the last 300 commits on
        `origin/main`.
        """

        descriptive = _module_frozenset("_DESCRIPTIVE_PROVENANCE_KEYS")
        self.assertNotIn("harness_digest", descriptive)
        self.assertIn("source_tree", descriptive)

    def test_the_differential_imports_the_digest_rather_than_reimplementing_it(self) -> None:
        source = DIFFERENTIAL.read_text(encoding="utf-8")
        self.assertIn("from harness_digest import harness_digest", source)


class TheEngineFingerprintStaysARebuildTriggerTests(unittest.TestCase):
    def test_build_inputs_excludes_the_python_harness(self) -> None:
        """The counterpart guard, and the reason a separate digest exists at all.

        Adding `scripts/` or `src/pokezero/**` to `build_inputs()` would close the
        same gap by forcing a 60-minute engine rebuild on every prose or test edit.
        That is the wrong trade and it would be an easy, plausible-looking future
        edit, so it is refused here rather than left to review.
        """

        offenders = []
        for path in fingerprint.build_inputs():
            rel = Path(path).resolve().relative_to(REPO).as_posix()
            if rel.startswith("scripts/") or rel.startswith("src/"):
                offenders.append(rel)
        self.assertEqual(
            offenders,
            [],
            "engine_build_fingerprint.build_inputs() gained Python harness files. "
            "That set decides when a REBUILD is needed; harness identity belongs in "
            "scripts/harness_digest.py, which costs nothing.",
        )

    def test_build_inputs_is_not_empty(self) -> None:
        """Anti-vacuity: the pin above passes over an empty list."""

        self.assertGreater(len(fingerprint.build_inputs()), 1)


# ---------------------------------------------------------------------------
# The drift, made assertable. This is the evidence the gap is real.
# ---------------------------------------------------------------------------

# The selector: every committed JSON under `reports/` or `docs/` (recursive) whose
# top-level `checkpoint_provenance.distinct` holds a provenance blob with BOTH a
# non-null `engine_fingerprint` and a non-null `source_commit`. `distinct` holds
# JSON STRINGS, not objects -- a scan that forgets to re-parse them reports zero
# carriers and concludes there is no drift, which is how the first pass at this
# measurement read.
#
# RE-DERIVED on this branch over the tree at `c69f033f`, not carried from an
# earlier measurement: 375 JSON files scanned, 82 carrying such a blob, 29 distinct
# engine fingerprints, SIX of which span more than one `source_commit`. An earlier
# statement of these figures was 26 and 5; it was measured on a different tree and
# is not what this corpus says.
#
# WHAT THE SIX MEAN. Each is a set of sweeps the engine fingerprint calls "the same
# build" that were produced at different commits, i.e. potentially by different
# harnesses. `5fa147ff` is the confirmed case: dev
# `strict:diverged_on_full_branch_set` = 1 on a fresh base build but absent from
# the C142 artifact at that same fingerprint, because the counter was added to the
# differential after C142's sweeps ran.
#
# WHY IT IS AN ALLOWLIST AND NOT A BAN. These six are history and cannot be
# re-measured; the assertion is that the set does not GROW unnoticed. A new sweep
# that lands under an existing fingerprint at a new commit either extends a group
# or creates a seventh, and either is red.
#
# The bump procedure, which is the sibling corpus pins' and is not optional:
#   1. run the selector below over BOTH trees -- this one and the base it came from;
#   2. confirm the set difference is exactly what the change adds, with NOTHING
#      removed (a group that silently shrinks is this pin's fail-open);
#   3. set the values to the measured ones;
#   4. confirm the pin is still LIVE by perturbing it and watching it fail.
# 82 -> 90 and 29 -> 31 (C152). RE-DERIVED by the four-step procedure above:
# `_provenance_blobs()` executed in a worktree of `origin/main` at `d20cf840` returns 82
# blobs / 29 fingerprints, and 90 / 31 here, with the set difference exactly C152's eight
# sweep artifacts and NOTHING removed. The two new fingerprints are:
#
#   bfdbe1c04876edcd1957e7a360c5086cfc7eae32ccf3ba0e71d137bd76df3990  -- the SHIPPING build,
#       74 patches at `d20cf840`. `reports/artifacts/c152_{head,h8_nowindow}_*` are the first
#       committed sweeps taken at the fingerprint that ships; every earlier pair was taken at
#       `8e912b45544034e6` or older.
#   89797289f4a3b55561863fd61438ac4d1987e1dfd210b399debd2f337b5b3c53  -- a THROWAWAY
#       INSTRUMENTED build, the shipping tree plus two `eprintln!` blocks (one in
#       `residual_disjoint_bands`, one in `leftovers_slot_truncated`) and nothing else. It is
#       recorded rather than hidden because `reports/artifacts/c152_wide_census_*_sweep.json`
#       were taken with it, and a fingerprint that appears in a committed artifact must be
#       explicable. It is NOT reproducible from any committed tree, by design, and no
#       fidelity claim rests on it.
_EXPECTED_PROVENANCE_BLOBS = 90
_EXPECTED_DISTINCT_FINGERPRINTS = 31
_KNOWN_MULTI_COMMIT_FINGERPRINTS = {
    "07a3290d11ca14ecfa8c70f89a82a99e5bdc5a47d24136f740d54c59ab3122b4": frozenset(
        {
            "05aef35f5655cb2105de0008cc6bf9a31a496de6",
            "0867a3aa15681c02ba37e304cf65973b12e9098c",
            "6b6fb3688b82328341725bfbd4ed46122d9a0404",
            "aeaee2b1e3aa73a76c4054ae56e85dfd59269765",
        }
    ),
    "5fa147ffa325c8872d47fbe3645125af9dac2c94f3b84ca6dc8be5f96539d341": frozenset(
        {
            "4c0ded451db822d207cbf5d985652a3154c1d85c",
            "662d9db8717fb31e2dc51d78e720ea846432ac72",
            "ce962c6e7aba13f6243bea5ae248fad597b37a10",
            "e0a23e4ed337df8c371635ab25dd2ac402bbe8d6",
        }
    ),
    "fdbf59379399b94447c029d402d837b1738ec6e6bba4bfe8992a38fd30528875": frozenset(
        {
            "b885037db4085f4bd7689172d5059a3c8eb1c448",
            "d27316b6f8564b29363c8a78d1f21a8d0be9738b",
            "dc6e1e197a54a734934cb04cc1b7986835a80ee4",
            "f876803e2ab5cdaf31a36849fe112062c77131f6",
        }
    ),
    "44ee1430708cbb55033f5c7f1234b4bf9699009e6ba6d9a972ba442df615d652": frozenset(
        {
            "3687d205f6f2ff43f88f080767bc5151ac27bd2a",
            "6d390acb4c38f73226ba70a11b05f6975179fe79",
        }
    ),
    "599c68a31e3734726481696bad99fb1f4eff5c463b20c6b3ff8510c3f86b00c1": frozenset(
        {
            "7d48330967b0f60819bec7c2689d125e50785b22",
            "87dd95c22dc1fff72e547e535c4ddcb623bc49ae",
        }
    ),
    "de29e3dc79c80659859fbe09ca9fc45dc9bc444bc8fb38cbe44bfa569259e840": frozenset(
        {
            "1c94f0714a9e693b3b9131865992a2bf11e21da0",
            "abbaff9a99c35b586fca581c02687a7f57695ee8",
        }
    ),
}

# ⚠ THE TRIPWIRE FIRED, 2026-08-08 (C152), which is what it was for. It read ZERO,
# because nothing had ever written a `harness_digest` when #1196 added the field.
# C152's eight sweeps are the first artifacts taken with it, and this is the review
# moment the comment below asked for: each of the eight carries EXACTLY ONE distinct
# harness digest, `e3459e1f2ce334848c2734ff8729627506032d204c72f0f1d3cf79256c9a7c85`,
# which is the value `harness_digest.harness_digest()` returns at `d20cf840` over a
# 73-file import closure -- re-derived, not copied from the artifacts it is checking.
#
# The original note, kept because it is the reason this number exists: this is a
# TRIPWIRE, not the assertion that carries the weight -- that one is
# `TheDifferentialStampsItTests::test_checkpoint_provenance_records_the_harness_digest`,
# which pins the producer.
_EXPECTED_HARNESS_STAMPED_ARTIFACTS = 8


def _provenance_blobs() -> list[tuple[str, dict]]:
    """(artifact, provenance blob) for every committed sweep-shaped JSON."""

    found: list[tuple[str, dict]] = []
    for pattern in ("reports/**/*.json", "docs/**/*.json"):
        for path in sorted(glob.glob(os.fspath(REPO / pattern), recursive=True)):
            try:
                loaded = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(loaded, dict):
                continue
            provenance = loaded.get("checkpoint_provenance")
            if not isinstance(provenance, dict):
                continue
            for entry in provenance.get("distinct") or []:
                if isinstance(entry, str):
                    try:
                        entry = json.loads(entry)
                    except json.JSONDecodeError:
                        continue
                if not isinstance(entry, dict):
                    continue
                if entry.get("engine_fingerprint") and entry.get("source_commit"):
                    found.append((os.path.relpath(path, REPO), entry))
    return found


def _fingerprint_to_commits() -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for _, blob in _provenance_blobs():
        grouped[blob["engine_fingerprint"]].add(blob["source_commit"])
    return dict(grouped)


class TheDriftIsAssertedNotMerelyDetectableTests(unittest.TestCase):
    def test_the_selector_finds_the_expected_number_of_provenance_blobs(self) -> None:
        # Every assertion below is a loop, and a loop over nothing passes.
        self.assertEqual(len(_provenance_blobs()), _EXPECTED_PROVENANCE_BLOBS)

    def test_the_distinct_fingerprint_count_is_pinned(self) -> None:
        self.assertEqual(
            len(_fingerprint_to_commits()),
            _EXPECTED_DISTINCT_FINGERPRINTS,
            "the set of engine fingerprints in the committed corpus changed size; "
            "re-derive per the procedure above _KNOWN_MULTI_COMMIT_FINGERPRINTS",
        )

    def test_the_set_of_fingerprints_spanning_several_commits_is_exactly_the_known_six(
        self,
    ) -> None:
        observed = {fp: commits for fp, commits in _fingerprint_to_commits().items() if len(commits) > 1}
        self.assertEqual(
            {fp: frozenset(commits) for fp, commits in observed.items()},
            _KNOWN_MULTI_COMMIT_FINGERPRINTS,
            "an engine fingerprint now spans a different set of source commits than "
            "the six historical groups. A new sweep landing under an existing "
            "fingerprint at a new commit is exactly the harness drift this pin "
            "exists to surface: record the group here and state which harness "
            "produced which artifact.",
        )

    def test_the_confirmed_case_is_still_in_the_corpus(self) -> None:
        """Anti-vacuity for the pin above: the dict could be edited to match anything.

        `5fa147ff` is the group with live evidence attached -- C142's artifacts and a
        fresh base build at the same fingerprint disagree on
        `strict:diverged_on_full_branch_set`. If it leaves the corpus the argument for
        this whole module has to be restated, not silently dropped.
        """

        grouped = _fingerprint_to_commits()
        confirmed = "5fa147ffa325c8872d47fbe3645125af9dac2c94f3b84ca6dc8be5f96539d341"
        self.assertIn(confirmed, grouped)
        self.assertGreater(len(grouped[confirmed]), 1)
        artifacts = {name for name, blob in _provenance_blobs() if blob["engine_fingerprint"] == confirmed}
        self.assertIn(os.path.join("reports", "artifacts", "c142_base_dev_sweep.json"), artifacts)

    def test_the_harness_stamped_artifact_count_is_pinned(self) -> None:
        stamped = {name for name, blob in _provenance_blobs() if blob.get("harness_digest")}
        self.assertEqual(
            len(stamped),
            _EXPECTED_HARNESS_STAMPED_ARTIFACTS,
            "a committed artifact now carries a harness_digest. Bump "
            "_EXPECTED_HARNESS_STAMPED_ARTIFACTS and confirm the artifact carries "
            f"exactly one. Found: {sorted(stamped)}",
        )

    def test_no_artifact_mixes_two_harnesses(self) -> None:
        """Vacuous today by construction, live the moment the first stamped sweep lands.

        A merged report whose shards came from two instruments describes neither. The
        resume guard refuses that on `--resume`; `--merge-from` does not compare
        resume identity, so the artifact is where it has to be caught.
        """

        per_artifact: dict[str, set[str]] = defaultdict(set)
        for name, blob in _provenance_blobs():
            if blob.get("harness_digest"):
                per_artifact[name].add(blob["harness_digest"])
        for name, digests in sorted(per_artifact.items()):
            with self.subTest(artifact=name):
                self.assertEqual(
                    len(digests), 1, f"{name} was produced by {len(digests)} harnesses"
                )


# ---------------------------------------------------------------------------
# Reachability. A pin nothing consumes is not a pin.
# ---------------------------------------------------------------------------

WORKFLOW = REPO / ".github" / "workflows" / "engine-fidelity-gates.yml"
_JOB_NAME = "harness-provenance"


def _job_block(name: str) -> str:
    """The YAML text of one top-level job, without importing a YAML parser.

    `setup-python` on a bare checkout has no PyYAML, and installing one to read a
    file this test could read directly would be the kind of dependency that turns a
    2-second unconditional job into a skippable one.
    """

    text = WORKFLOW.read_text(encoding="utf-8")
    marker = f"\n  {name}:\n"
    rest = text[text.index(marker) + len(marker) :]
    lines: list[str] = []
    for line in rest.splitlines():
        # A job's body is indented four spaces or more. Anything at exactly two --
        # the next job's key, or the comment block introducing it -- ends this one.
        if line.startswith("  ") and not line.startswith("   "):
            break
        lines.append(line)
    return "\n".join(lines)


def _guard_body(block: str, needle: str) -> str:
    """The shell `if ...; then ... fi` region containing `needle`.

    Used to assert that a guard's failure arm actually exits nonzero. A structural
    check that the guard EXISTS says nothing about whether it does anything.
    """

    lines = block.splitlines()
    for index, line in enumerate(lines):
        if needle not in line:
            continue
        body = [line]
        for following in lines[index + 1 :]:
            body.append(following)
            if following.strip() == "fi":
                break
        return "\n".join(body)
    raise AssertionError(f"no guard containing {needle!r} in this block")


class ThePinIsReachableFromTheRequiredContextTests(unittest.TestCase):
    """This repo has found FOUR inert pins, the last a CI job whose result nothing
    consumed: `seed-registry pass 8s` beside `gate-status pass 3s`, the reporter
    returning before the job it reported on could have mattered. Branch protection
    requires exactly one context, so a job absent from `gate-status`'s `needs` runs,
    goes red, and changes nothing. These four pins are what make that structural
    rather than a thing someone remembered."""

    def test_the_workflow_defines_the_job(self) -> None:
        self.assertIn(f"\n  {_JOB_NAME}:\n", WORKFLOW.read_text(encoding="utf-8"))

    def test_the_job_is_unconditional(self) -> None:
        """No `if:` at ANY depth, so `skipped` is not reachable and `gate-status`
        needs no case analysis for it -- which is what lets that check demand the
        positive rather than exclude a stale list of negatives.

        THREE REVISIONS, each defeated by the next mutation, recorded because the
        pattern is the point: an assertion about YAML written as a string or indent
        search keeps having a shape the YAML can slip around.

          1. `"\\n    if:" not in block` -- defeated by `if:` as the job's FIRST key,
             which has no leading newline inside the block slice.
          2. A 4-space key scan splitting on ":" -- defeated TWO ways, both found in
             review: `if : false` (space before the colon, so the key parsed as
             `"if "` and did not match `"if"`, while `yaml.safe_load` reads the key
             as `if`), and a STEP-level `if: false` at 8 spaces, invisible to a scan
             that only looked at 4. The second is the worse of the two -- a job whose
             every step skips still reports `success`, so `gate-status` is satisfied
             and the pin is neutered while the whole workflow stays green.
          3. This one. Every key at every depth, normalized before comparison.

        And a THIRD defeat found here rather than in review, of the same family:
        `continue-on-error: true`. It is documented to stop a failing job from
        failing the workflow, so the pin could go red and the required context still
        pass. It is forbidden at every depth for the same reason `if` is -- there is
        no legitimate use of either key anywhere in this job.
        """

        keys: list[str] = []
        for line in _job_block(_JOB_NAME).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            # A step is a list item, so its first key arrives as `- name: ...`.
            if stripped.startswith("- "):
                stripped = stripped[2:]
            # Normalize BEFORE splitting, not after: `if : false` must key as `if`.
            # Quotes stripped too -- `"if": false` is a valid YAML mapping key that
            # parses as `if`, and without this it keyed as `"if"` and slipped the
            # check. Found by review as a sixth defeat attempt, outside the declared
            # threat model but a genuine one-line hardening.
            keys.append(stripped.split(":")[0].strip().strip("\"'"))

        self.assertNotIn("if", keys, "the job or one of its steps became skippable")
        self.assertNotIn(
            "continue-on-error",
            keys,
            "a failing pin would no longer fail the job, so gate-status would still pass",
        )
        self.assertNotIn("needs", keys, "the job now depends on another job and can be skipped")
        # Anti-vacuity, and it has to name a key at each depth the scan claims to
        # reach -- a scan that silently stopped seeing step-level keys would pass an
        # anti-vacuity check that only asked for `runs-on`.
        self.assertIn("runs-on", keys, "anti-vacuity: the job-level key scan found nothing")
        self.assertIn("name", keys, "anti-vacuity: the step-level key scan found nothing")
        self.assertIn("run", keys, "anti-vacuity: the step-body key scan found nothing")

    def test_the_job_actually_runs_this_module(self) -> None:
        """The `Ran N tests` and `^OK$` guards read a LOG, and a log can be forged.

        Replacing the `run:` body with `echo 'Ran 27 tests'; echo OK` satisfies both
        greps. Pinning the invocation is what makes them mean anything -- the fourth
        way I could find to neuter this job while leaving it green.
        """

        block = _job_block(_JOB_NAME)
        self.assertIn(
            "python -m unittest tests.test_harness_digest_provenance",
            block,
            "the job no longer runs this module; its log guards assert nothing",
        )
        self.assertIn("python scripts/harness_digest.py --print", block)
        # Both log guards must exit nonzero, for the same reason gate-status's
        # branch must: `exit 1` -> `exit 0` is a one-character edit that leaves the
        # grep, the message and the whole shape of the guard intact while the job
        # passes over a shrunk or forged suite.
        for needle in ("Ran 28 tests", "'^OK$'"):
            with self.subTest(guard=needle):
                self.assertIn(
                    "exit 1",
                    _guard_body(block, needle),
                    f"the {needle} guard no longer exits nonzero",
                )

    def test_gate_status_needs_the_job_and_consumes_its_result(self) -> None:
        block = _job_block("gate-status")
        needs = next(line for line in block.splitlines() if line.strip().startswith("needs:"))
        self.assertIn(_JOB_NAME, needs, "the job is not in the required context's needs")
        self.assertIn(f"needs.{_JOB_NAME}.result", block, "the result is not read")
        self.assertIn(
            '[ "${HARNESS}" != "success" ]',
            block,
            "gate-status reads the result but does not fail on it",
        )
        # ...and the branch must actually FAIL. Found by mutation while hardening
        # the pins above: flipping this one `exit 1` to `exit 0` leaves every
        # structural assertion satisfied -- the job is needed, the result is read,
        # the comparison is present -- and the required context goes green over a
        # red pin. Testing for the presence of a check is not testing that the
        # check does anything, which is the whole four-inert-pins lesson one level
        # further in.
        self.assertIn(
            "exit 1",
            _guard_body(block, '[ "${HARNESS}" != "success" ]'),
            "the harness branch of gate-status no longer exits nonzero",
        )

    def test_the_workflow_test_count_guard_matches_this_module(self) -> None:
        """The guard lives in YAML, so no local `unittest` or `pytest` run can see it.

        A module that grows without its `Ran N tests` guard following is how an
        approved PR has reddened CI in this repo before. Making the module check its
        own guard removes that failure mode for this module specifically -- it cannot
        be forgotten, because forgetting it is a local failure.
        """

        loaded = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        count = loaded.countTestCases()
        self.assertIn(
            f"Ran {count} tests",
            _job_block(_JOB_NAME),
            f"this module now has {count} tests; the workflow's exact grep guard for "
            f"{_JOB_NAME} still names a different number",
        )


if __name__ == "__main__":
    unittest.main()
