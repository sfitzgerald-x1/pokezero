"""Re-derive the ledger's "never fired" lists from the committed artifacts.

WHY THIS MODULE EXISTS. `reports/c138_known_gaps_ledger.md` has been wrong about a
"never fired" counter four times, and every one of them was a *prose* assertion that no
check re-derived:

  * H14 said `skip:strict_all_branches_lossy` "has never fired". It fires at 2 in
    `reports/c26_structural_probe_report.json` and `c27_structural_probe_report.json`, at
    4 in C141's final-holdout sweep, and at **372** in `reports/c32_fail_diagnosis.json`
    under the differently-named field `coverage_diagnosis.coverage_reducing_skips.*`.
    Corrected by #1163.
  * The "Never-fired static counters" list said 10; the measured figure is 9. Corrected
    by #1162.
  * H11 asserted "no written cause anywhere in `reports/`" against a file that was
    already committed. Corrected by #1165.
  * H13 asserted that `self_moveset_mismatch`, `transform_unexpressible` and
    `status_unsupported` "never fire in either window". All three have fired, and
    `self_moveset_mismatch` fired at 75 dev / 24 holdout on the **byte-identical seed
    windows** (`seeds.min/max` 19000000-19000199 and 19100000-19100199) across 27
    committed sweep artifacts, c121 through c133. Corrected by the PR that adds this
    module. See `reports/c146_negative_claim_audit.md`.

Prose cannot prevent the fifth. These pins can, because they compute the partition
instead of restating it.

THREE DESIGN CHOICES, each taken because its absence caused one of the four errors:

  1. **The corpus is the whole of `reports/` AND `docs/`, recursively.** H15's
     "6 of the 19 divergence_class values have ever fired" was measured over
     `reports/artifacts/` alone, where it is true. Repo-wide it is 7 --
     `limit:world_sample_drag_target` fires at 5 in `reports/c10_encore_differential.json`
     and at 4 in `reports/c26_structural_probe_report.json`, neither of which is under
     `reports/artifacts/`. Scoping a glob to one directory and reporting the result as
     repo-wide is this program's recurring defect, so the selector here is deliberately
     wider than any single claim needs.
     `docs/` is not decorative: `docs/audit_artifacts/**` is where
     `transform_unexpressible` reaches 208 and `self_moveset_mismatch` reaches 2560.

  2. **A name is matched anywhere in the leaf's PATH, not only as a counter key**, and
     additionally through the `{"counter": "<name>", "rows": N}` shape. The C32/C43
     precedent is the whole lesson: an audit keyed on `counters.skip:...` alone declares
     `strict_all_branches_lossy` never-fired and is wrong. Both shapes carry a dedicated
     anti-vacuity pin below, so a regression in either matcher goes red rather than quiet.

  3. **The partitions are asserted as exact SET EQUALITY, never as "these are absent".**
     An absence-only loop passes when the scanner returns nothing, which is the fail-open
     that a broken selector produces. The fired set must also be exactly right, so a
     counter that starts firing turns this module red instead of silently widening.

The reason lists themselves are derived from source by AST at test time, not transcribed.
A new `EngineWorldUnsupported` reason, or a new `classify_divergence` return site, moves
the count and fails the count pin -- which is the point: a refusal added without a ledger
row is exactly how §3.5 goes stale.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Directories that never hold committed measurement artifacts, and one (`third_party`)
# that is gitignored and regenerated, so walking it is both slow and non-deterministic.
_SKIP_DIRS = frozenset(
    {
        ".git",
        "third_party",
        "node_modules",
        "target",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "dist",
        "build",
    }
)

# The two trees walked. Verified at the time of writing by scanning ALL 379 committed
# JSON outside `third_party/` -- every nonzero counter-shaped leaf for every name below
# lives under one of these two. `runs/`, `evals/`, `scenarios/`, `checkpoints/`,
# `schemas/` and `tests/data/` hold JSON and contribute no counter evidence.
_CORPUS_TREES = ("reports", "docs")

# MEASURED, not computed: 347 == 267 under `reports/` + 80 under `docs/`. The sum is
# recorded only to make the number legible; it was read off the selector, and the two
# subtotals must not be used to sanity-check each other (see
# `_EXPECTED_SWEEP_ARTIFACTS` in `tests/test_boundary_verdict_partition.py` for the same
# trap). Bumping this number is only correct after confirming the set DIFFERENCE against
# the base tree is exactly what the PR adds, with nothing removed -- a member that
# vanishes is this pin's fail-open, and pure addition masks it.
_EXPECTED_COUNTER_ARTIFACTS = 347

# ---------------------------------------------------------------------------
# The taxonomies, derived from source rather than transcribed.
# ---------------------------------------------------------------------------

_EXPECTED_WORLD_UNSUPPORTED_REASONS = 40
_EXPECTED_DIVERGENCE_CLASSES = 19

# The 10 of 40 `EngineWorldUnsupported` reasons with nonzero recorded evidence somewhere
# in the corpus. Note what this list is NOT: it is not "fires in the c136 windows", which
# is only four of them (`encore_move_unknown`, `materialization_blocker`,
# `self_request_state_unsupported`, `volatile_unsupported`). The other six fired in
# earlier eras or in the `docs/audit_artifacts` search grids, and a closed exit is not a
# never-fired one. Conflating those two readings is precisely the H13 defect.
_FIRED_WORLD_UNSUPPORTED = frozenset(
    {
        "encore_move_unknown",
        "materialization_blocker",
        "payload_malformed",
        "pending_baton_pass",
        "self_moveset_mismatch",
        "self_request_state_unsupported",
        "status_unsupported",
        "substitute_health_unknown",
        "transform_unexpressible",
        "volatile_unsupported",
    }
)

# 7 of the 19 static `classify_divergence` return sites, against H15's "only 6". The
# seventh is `limit:world_sample_drag_target`; H15 lists it among "strict-path classes
# the program has simply never produced".
_FIRED_DIVERGENCE_CLASSES = frozenset(
    {
        "component_extra_in_engine",
        "component_magnitude",
        "component_mismatch",
        "component_missing_in_engine",
        "limit:roll_divergent_lethality",
        "limit:world_sample_drag_target",
        "roll_scaled_component",
    }
)

# §3.5's "Never-fired static counters (9)", verified here over a corpus wider than the
# one §3.5 used (it searched `reports/` only, and quoted 260 files; this is 347 across
# `reports/` and `docs/`). All nine survive the wider glob. `engine_error` is also one of the two
# unexercised terms of the shipped verdict identity, alongside `skip:rump_branch_set`; C144
# stated it as four-term and the code is five-term, annotated at H14.
_NEVER_FIRED_STATIC_COUNTERS = (
    "abort:no_legal_action",
    "skip:no_action_candidates",
    "no_constructible_candidate",
    "no_damage_rolls",
    "BranchLegalRollError",
    "engine_error",
    "world_prestate_mismatch:side_conditions",
    "mapper_lossy",
    "no_usable_branch",
)

# NOT one of §3.5's nine, and kept separate so that list keeps meaning nine.
# `skip:rump_branch_set` is the FIFTH term of the shipped verdict identity
# (`cert_sweep_readout.py:1451,1611`) -- H14 states that identity as four-term, which C146
# annotates in place, and this is its second never-exercised term alongside `engine_error`.
# It was asserted in §3.3 and held by no pin, which is exactly the class of claim this
# module exists to close: a verified negative nothing re-derives is an asserted one waiting
# to rot. Measured 0 across all 347.
_NEVER_FIRED_VERDICT_IDENTITY_TERMS = ("rump_branch_set",)

# §3.5's "7 of 8 unmappable_choice reasons unobserved", plus the eighth as the
# anti-vacuity control on the same taxonomy.
_NEVER_FIRED_UNMAPPABLE_CHOICE = (
    "no_candidate_row",
    "blank_move_id",
    "hidden_power_ambiguous",
    "move_not_in_engine_set",
    "blank_switch_species",
    "switch_species_not_in_party",
    "unmappable_choice:unknown_kind",
)
_FIRED_UNMAPPABLE_CHOICE = ("struggle_not_submittable",)

# §3.5's "Never-fired dynamic families (6)". Matched by key PREFIX, because the whole
# point of these is that the suffix is an interpolated exception or choice name.
_NEVER_FIRED_DYNAMIC_FAMILIES = (
    "skip:no_materialization:",
    "skip:world_error:",
    "strict:branch_events_error:",
    "engine_error:",
    "engine_error_choice:",
    "world_prestate_mismatch:weather_",
)


def _world_unsupported_reasons() -> set[str]:
    """Every literal first argument to a `raise EngineWorldUnsupported(...)`."""
    tree = ast.parse((REPO / "src/pokezero/engine_world.py").read_text(encoding="utf-8"))
    reasons: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        func = node.exc.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "EngineWorldUnsupported" or not node.exc.args:
            continue
        first = node.exc.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            reasons.add(first.value)
    return reasons


def _divergence_classes() -> set[str]:
    """Every static class `classify_divergence` can return.

    A `"prefix:" + payload` return contributes its literal prefix, because that is what
    appears in the counter key ahead of the dynamic component list.
    """
    tree = ast.parse(
        (REPO / "scripts/engine_transition_differential.py").read_text(encoding="utf-8")
    )
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "classify_divergence"
    )
    classes: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            classes.add(value.value)
        elif isinstance(value, ast.BinOp) and isinstance(value.left, ast.Constant):
            literal = value.left.value
            classes.add(literal.rstrip(":").split(":%s")[0])
    return classes


# ---------------------------------------------------------------------------
# The scanner.
# ---------------------------------------------------------------------------

# Prefixes stripped before comparing a string FIELD to a reason name, so that C43's
# `{"counter": "world_unsupported:self_moveset_mismatch", "rows": 5058}` matches
# `self_moveset_mismatch`. Applied repeatedly: `skip:world_unsupported:x` -> `x`.
_KEY_PREFIXES = (
    "skip:",
    "strict:",
    "abort:",
    "divergence_class:",
    "world_unsupported:",
    "unmappable_choice:",
    "world_error:",
    "no_materialization:",
)


def _strip_prefixes(text: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _KEY_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix) :]
                changed = True
    return text


def _is_nonzero_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0


@functools.lru_cache(maxsize=1)
def counter_artifacts() -> tuple[str, ...]:
    """Every committed JSON under `reports/` or `docs/`, recursively."""
    found: list[str] = []
    for tree in _CORPUS_TREES:
        for root, dirs, files in os.walk(REPO / tree):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for name in files:
                if name.endswith(".json"):
                    found.append(os.path.relpath(os.path.join(root, name), REPO))
    return tuple(sorted(found))


@functools.lru_cache(maxsize=1)
def _parsed_corpus() -> tuple[tuple[str, object], ...]:
    """`(artifact, document)` for every member. Cached: the pins below scan the corpus
    several times each, and re-reading 347 files per test made the module slow enough to
    invite a CI author to drop it.

    **Nothing is skipped.** An earlier revision swallowed `OSError`/`JSONDecodeError` and
    continued, which is not live -- all 347 parse today -- but is precisely the vacuous-green
    failure the witness pins below exist to prevent: an artifact going unparseable would
    silently shrink the evidence base and every absence assertion would get *easier*. A file
    that cannot be read is a red gate, not one fewer haystack.
    """
    loaded: list[tuple[str, object]] = []
    unreadable: list[str] = []
    for artifact in counter_artifacts():
        try:
            loaded.append(
                (artifact, json.loads((REPO / artifact).read_text(encoding="utf-8")))
            )
        except (OSError, json.JSONDecodeError) as exc:
            unreadable.append(f"{artifact}: {type(exc).__name__}: {exc}")
    if unreadable:
        raise AssertionError(
            "committed JSON in the counter-artifact corpus could not be read, so the "
            "evidence base for every absence pin below is incomplete. Fix or remove the "
            "file; do not let it drop out silently: " + "; ".join(unreadable)
        )
    return tuple(loaded)


def _token(name: str) -> re.Pattern[str]:
    return re.compile(r"(^|[^A-Za-z0-9_])" + re.escape(name) + r"($|[^A-Za-z0-9_])")


def _evidence_in(doc: object, patterns: dict[str, re.Pattern[str]]) -> dict[str, tuple[str, object]]:
    """First `(path, value)` witness per name: a nonzero number named by this document.

    Two admitted shapes, and no third:
      * the name appears as a token anywhere in the leaf's dotted PATH and the leaf is a
        nonzero number -- this covers `counters.skip:world_unsupported:X`,
        `coverage_diagnosis.coverage_reducing_skips.X`, `divergence_classes.X` and the
        `world_failure_reasons.X: <detail>` expansions in one rule;
      * a mapping has a string FIELD equal to the name after prefix-stripping AND a
        nonzero numeric sibling -- C43's ranked-decomposition shape.

    A name merely mentioned inside prose is NOT evidence. That exclusion is load-bearing:
    admitting it made `no_usable_branch` and `BranchLegalRollError` read as fired, off
    `reports/c9_decomposition.json`'s `"basis"` narration and a c17 sentence, and both are
    genuinely never-fired.
    """
    out: dict[str, tuple[str, object]] = {}

    def visit(node: object, path: str) -> None:
        if isinstance(node, dict):
            numbers = [(k, v) for k, v in node.items() if _is_nonzero_number(v)]
            if numbers:
                strings = [v for v in node.values() if isinstance(v, str)]
                for name in patterns:
                    if name in out:
                        continue
                    for text in strings:
                        if _strip_prefixes(text) == name:
                            out[name] = (f"{path}{{={text}}}", numbers[0][1])
                            break
            for key, value in node.items():
                visit(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{path}[{index}]")
        elif _is_nonzero_number(node):
            for name, pattern in patterns.items():
                if name not in out and pattern.search(path):
                    out[name] = (path, node)

    visit(doc, "")
    return out


def scan(names) -> dict[str, list[tuple[str, str, object]]]:
    """`{name: [(artifact, path, value), ...]}` over the whole corpus."""
    patterns = {name: _token(name) for name in names}
    found: dict[str, list[tuple[str, str, object]]] = {}
    for artifact, doc in _parsed_corpus():
        for name, (path, value) in _evidence_in(doc, patterns).items():
            found.setdefault(name, []).append((artifact, path, value))
    return found


def scan_prefixes(prefixes) -> dict[str, list[tuple[str, str, object]]]:
    """As `scan`, but a key SEGMENT must start with the prefix."""
    found: dict[str, list[tuple[str, str, object]]] = {}
    for artifact, doc in _parsed_corpus():
        stack: list[tuple[object, tuple[str, ...]]] = [(doc, ())]
        while stack:
            node, path = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    stack.append((value, path + (str(key),)))
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    stack.append((value, path + (f"[{index}]",)))
            elif _is_nonzero_number(node):
                for prefix in prefixes:
                    if any(seg.startswith(prefix) for seg in path):
                        found.setdefault(prefix, []).append(
                            (artifact, ".".join(path), node)
                        )
    return found


class CorpusIsWhatWeThinkItIsTests(unittest.TestCase):
    def test_the_counter_artifact_corpus_is_exactly_the_expected_size(self) -> None:
        # Every loop below is a loop, and a loop over nothing passes. A floor is not
        # enough either: an artifact leaving the corpus must be red, not slack.
        found = counter_artifacts()
        self.assertEqual(
            len(found),
            _EXPECTED_COUNTER_ARTIFACTS,
            "the committed counter-artifact corpus changed size. If JSON was added under "
            "reports/ or docs/, bump _EXPECTED_COUNTER_ARTIFACTS after confirming the set "
            "difference against the base tree is exactly the additions, with nothing "
            f"removed. Found {len(found)} files.",
        )

    def test_the_corpus_spans_both_trees(self) -> None:
        # A selector that silently loses `docs/` still passes every absence pin, and
        # `docs/audit_artifacts/**` is where two of H13's three named reasons fire.
        found = counter_artifacts()
        self.assertTrue(any(f.startswith("reports/") for f in found))
        self.assertTrue(
            any(f.startswith(os.path.join("docs", "audit_artifacts")) for f in found),
            "docs/audit_artifacts left the corpus; H13's refutation lives partly there",
        )


class TaxonomySizesAreDerivedFromSourceTests(unittest.TestCase):
    def test_the_world_unsupported_reason_count_is_pinned(self) -> None:
        reasons = _world_unsupported_reasons()
        self.assertEqual(
            len(reasons),
            _EXPECTED_WORLD_UNSUPPORTED_REASONS,
            "the EngineWorldUnsupported reason taxonomy changed size. Add or retire the "
            "matching row in reports/c138_known_gaps_ledger.md §3.5, then bump this pin. "
            f"Reasons: {sorted(reasons)}",
        )

    def test_the_divergence_class_count_is_pinned(self) -> None:
        classes = _divergence_classes()
        self.assertEqual(
            len(classes),
            _EXPECTED_DIVERGENCE_CLASSES,
            "classify_divergence's static return sites changed count; H15 quotes this "
            f"number. Classes: {sorted(classes)}",
        )


class NeverFiredPartitionsTests(unittest.TestCase):
    def test_the_world_unsupported_fired_set_is_exactly_pinned(self) -> None:
        reasons = _world_unsupported_reasons()
        found = scan(reasons)
        fired = {name for name in reasons if name in found}
        # Set EQUALITY in both directions. A newly-firing reason must be a failure, and
        # so must a reason that stops being found -- the latter means the scanner broke.
        self.assertEqual(
            fired,
            set(_FIRED_WORLD_UNSUPPORTED),
            "the world_unsupported fired/never-fired partition moved.\n"
            f"  newly fired: {sorted(fired - set(_FIRED_WORLD_UNSUPPORTED))}\n"
            f"  no longer found: {sorted(set(_FIRED_WORLD_UNSUPPORTED) - fired)}\n"
            "Update H13 and §3.5 in reports/c138_known_gaps_ledger.md in the same commit.",
        )

    def test_h13s_three_named_reasons_have_all_fired(self) -> None:
        # The specific refutation, pinned by name so the claim cannot come back as prose.
        found = scan(
            ["self_moveset_mismatch", "transform_unexpressible", "status_unsupported"]
        )
        for name in ("self_moveset_mismatch", "transform_unexpressible", "status_unsupported"):
            with self.subTest(reason=name):
                self.assertIn(
                    name,
                    found,
                    f"{name} is recorded in the corpus with a nonzero count; H13 called it "
                    "never-fired and that was the error this pin exists to prevent",
                )

    def test_the_divergence_class_fired_set_is_exactly_pinned(self) -> None:
        classes = _divergence_classes()
        found = scan(classes)
        fired = {name for name in classes if name in found}
        self.assertEqual(
            fired,
            set(_FIRED_DIVERGENCE_CLASSES),
            "the divergence_class fired/never-fired partition moved. H15 quotes "
            f"'6 of 19'; the repo-wide figure is {len(_FIRED_DIVERGENCE_CLASSES)} of "
            f"{len(classes)}.\n  newly fired: "
            f"{sorted(fired - set(_FIRED_DIVERGENCE_CLASSES))}\n"
            f"  no longer found: {sorted(set(_FIRED_DIVERGENCE_CLASSES) - fired)}",
        )

    def test_the_never_fired_static_counters_are_still_absent(self) -> None:
        # §3.5's nine, plus `skip:rump_branch_set` -- see its constant for why it is
        # tracked separately but asserted here.
        names = _NEVER_FIRED_STATIC_COUNTERS + _NEVER_FIRED_VERDICT_IDENTITY_TERMS
        self.assertEqual(len(names), 10, "the never-fired static list changed shape")
        found = scan(names)
        self.assertEqual(
            {name: found[name][:2] for name in found},
            {},
            "a never-fired static counter now has recorded evidence",
        )

    def test_seven_of_the_eight_unmappable_choice_reasons_are_absent(self) -> None:
        found = scan(_NEVER_FIRED_UNMAPPABLE_CHOICE + _FIRED_UNMAPPABLE_CHOICE)
        self.assertEqual(
            sorted(found),
            sorted(_FIRED_UNMAPPABLE_CHOICE),
            "the unmappable_choice partition moved; §3.5 says 7 of 8 are unobserved",
        )

    def test_the_six_dynamic_families_are_still_absent(self) -> None:
        found = scan_prefixes(_NEVER_FIRED_DYNAMIC_FAMILIES)
        self.assertEqual(
            {prefix: found[prefix][:2] for prefix in found},
            {},
            "a §3.5 'never-fired dynamic family' now has recorded evidence",
        )


class TheMatchersThemselvesAreExercisedTests(unittest.TestCase):
    """Anti-vacuity. Each absence pin above is only worth its matcher, so each of the
    two admitted evidence shapes is pinned against the exact artifact that motivated it.
    Without these, a matcher that stopped matching would turn every pin green."""

    def test_the_c32_differently_named_field_shape_is_matched(self) -> None:
        # `coverage_diagnosis.coverage_reducing_skips.<reason>` -- not a counter key.
        # This is the shape whose omission produced the H14 error.
        found = scan(["self_moveset_mismatch"])["self_moveset_mismatch"]
        witness = [
            (path, value)
            for artifact, path, value in found
            if artifact == "reports/c32_fail_diagnosis.json"
        ]
        self.assertEqual(
            witness,
            [("coverage_diagnosis.coverage_reducing_skips.self_moveset_mismatch", 5058)],
            "the path matcher no longer reaches c32's coverage_reducing_skips shape",
        )

    def test_the_c43_sibling_field_shape_is_matched(self) -> None:
        # `{"counter": "world_unsupported:transform_unexpressible", "rows": 23}`.
        found = scan(["transform_unexpressible"])["transform_unexpressible"]
        witness = [
            value
            for artifact, path, value in found
            if artifact == "reports/c43_coverage_shortfall_diagnosis.json"
        ]
        self.assertEqual(
            witness,
            [23],
            "the sibling-field matcher no longer reaches c43's ranked-decomposition shape",
        )

    def test_prose_alone_is_not_evidence(self) -> None:
        # The control on the other side: `no_usable_branch` appears in
        # `reports/c9_decomposition.json` and `c12_decomposition.json` inside a `"basis"`
        # narration next to a large unrelated number, and it is genuinely never-fired.
        # Admitting prose flipped it, and flipped `BranchLegalRollError` too.
        found = scan(["no_usable_branch", "BranchLegalRollError"])
        self.assertEqual(found, {}, "prose mentions are being counted as evidence again")


class H13sRefutationIsPinnedToTheWindowsTests(unittest.TestCase):
    """H13's claim was scoped "in either window", so the refutation is pinned on the
    windows' own terms: the c121/c133 pairs and the c136 pair are the SAME 200 seeds."""

    _WINDOWS = {
        "dev": (19000000, 19000199),
        "holdout": (19100000, 19100199),
    }
    _SELF_MOVESET_MISMATCH = "counters.skip:world_unsupported:self_moveset_mismatch"

    def _load(self, name: str) -> dict:
        return json.loads((REPO / "reports/artifacts" / name).read_text(encoding="utf-8"))

    def test_the_pre_and_post_closure_sweeps_share_one_seed_window(self) -> None:
        for window, (low, high) in self._WINDOWS.items():
            for stem in (
                "c121_a5_{}_sweep.json",
                "c133_withdrawn_switchcancel_{}_sweep.json",
                "c136_faintcancels_fix_{}_sweep.json",
            ):
                name = stem.format(window)
                with self.subTest(artifact=name):
                    seeds = self._load(name)["seeds"]
                    self.assertEqual((seeds["min"], seeds["max"], seeds["distinct"]),
                                     (low, high, 200))

    def test_self_moveset_mismatch_fired_in_both_windows_before_it_closed(self) -> None:
        # 75 dev / 24 holdout. These are the numbers H13 called zero.
        for window, expected in (("dev", 75), ("holdout", 24)):
            for stem in (
                "c121_a5_{}_sweep.json",
                "c133_withdrawn_switchcancel_{}_sweep.json",
            ):
                name = stem.format(window)
                with self.subTest(artifact=name):
                    counters = self._load(name)["counters"]
                    self.assertEqual(
                        counters.get("skip:world_unsupported:self_moveset_mismatch"),
                        expected,
                    )

    def test_the_closure_is_visible_as_a_closure_not_an_absence(self) -> None:
        # And it closed: 0 from the c134/c136 generation on, with the freed skips
        # reappearing as measured boundaries rather than vanishing. Dev:
        # 15432 -> 15503 is +71, against -75 self_moveset_mismatch and +4
        # substitute_health_unknown. That reconciliation is what makes this a fix.
        before = self._load("c133_withdrawn_switchcancel_dev_sweep.json")
        after = self._load("c136_faintcancels_fix_dev_sweep.json")
        self.assertEqual(before["counters"]["skip:world_unsupported:self_moveset_mismatch"], 75)
        self.assertEqual(
            after["counters"].get("skip:world_unsupported:self_moveset_mismatch", 0), 0
        )
        self.assertEqual(after["boundaries_measured"] - before["boundaries_measured"], 71)
        freed = 75 - (
            after["counters"]["limit:world_substitute_health_unknown"]
            - before["counters"]["limit:world_substitute_health_unknown"]
        )
        self.assertEqual(freed, 71, "the freed skips did not reappear as measured boundaries")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
