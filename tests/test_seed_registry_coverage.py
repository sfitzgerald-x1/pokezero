"""Re-derive the fidelity seed registry's coverage from the committed artifacts.

WHY THIS MODULE EXISTS. The seed registry in
`docs/engine_divergence_ledger_20260728.md` (§ "the seed space is therefore
partitioned by purpose") is prose. Nothing parsed it, and it was wrong about the
final-holdout row in two independent ways at `553cf2c3`:

  * The row said **"reserved, untouched"**. `reports/artifacts/c141_final_holdout_sweep.json`
    had already swept `19,200,060`-`19,200,259`, 200 games, 16,274 boundaries
    measured, and that artifact has been committed the whole time.
  * The row's recorded span was `19,200,000`-`19,200,199`, which is **not** what was
    swept. The sweep overran the registered end by 60 seeds and left the first 60
    registered seeds unswept.

Both defects are of the same kind the never-fired counter census (C146,
`tests/test_never_fired_counter_census.py`) was built for: a claim about artifacts
that no check re-derived FROM the artifacts. This module is the same instrument
pointed at the seed axis.

THE SELECTOR IS SHAPE-AGNOSTIC, AND THAT IS THE WHOLE POINT.

A selector keyed on `seeds.min` under `reports/artifacts/` is the natural one to
reach for -- 80 of the 93 fidelity-space artifacts answer to it. It is also the one
that produced the false claim "everything at or above `19,200,260` is virgin",
because its highest seed is exactly `19,200,259` and it cannot see
`reports/c73_eight_hundred_game_sweep.json`, which sweeps
`19,500,000`-`19,500,799`, records `run.seed_start` rather than `seeds.min`, and
lives one directory up. `TheNaiveSelectorIsTheRecordedMistakeTests` below pins that
miss against the real tree, so the lesson cannot be re-learned by hand a third time.

FOUR span shapes are actually in the corpus, measured, not assumed:

  1. `{"seeds": {"min": ..., "max": ...}}`            -- 80 files under `reports/artifacts/`,
                                                          15 directly under `reports/` (16 if
                                                          any nested `min`/`max` pair counts),
                                                          0 under `docs/`
  2. `{"run":     {"seed_start": ..., "games": ...}}` -- `c72`, `c73`
  3. `{"sample":  {"seed_start": ..., ...}}`, closed by `seed_end` OR by `games`
                                                      -- `c82` carries `seed_end`;
                                                         `c83` and `c86` carry only `games`
  4. `{"windows": {"dev": {"seed_start": ..., "games": ...}, "holdout": {...}}}`
                                                      -- `c147_g33b_gate_reach`

Shape 3 is why the extractor computes an end from a game count rather than requiring
one: three of these files never state their last seed. `c73` is the same case and is
the one that mattered -- its `19,500,799` appears in no JSON anywhere.

So `_seed_intervals` does not enumerate shapes at all. It walks every dict in the
document and takes:

  * any `min`/`max` integer pair, at any depth;
  * any integer under a key containing both `seed` and `start`, closed with a
    sibling `seed_end` or a sibling `games` count when one is present;
  * every other integer under a key containing `seed`, as a one-seed interval --
    which is how `repros[].seed` and `per_game[].seed` are covered without naming
    them.

A fifth shape invented next month is therefore covered on arrival, and the pin
fails loudly rather than silently narrowing. That is the opposite bargain from the
naive selector, and it is deliberate: this repo's recurring defect is a scoped glob
reported as a repo-wide result.

WHAT IS *NOT* ASSERTED HERE, ON PURPOSE. This module says nothing about whether any
band may be swept in future, and nothing about the ledger's status *words*. The
status column is prose with real nuance -- the validation holdout is described as
"reserved (C116 Phase 1)" and is nonetheless swept on every fix branch, so a rule of
the form "reserved implies no artifact" would be false on the first row it touched.
That is exactly the trap the previous ledger-checker attempt fell into: it encoded a
rule its author inferred rather than measured. The rule pinned here is measured, and
it is about seeds only: **every fidelity seed in the committed record lies inside a
registered band, and every registered band has a committed witness.**

It also says nothing about MULTIPLICITY. This is containment, not counting: a second
sweep of `19,200,060`-`19,200,259` would sit inside a registered band and pass every
assertion in this file. The "the final holdout appears in exactly ONE measurement"
invariant belongs to `#1122` -- `_reject_unguarded_final_holdout` in
`scripts/engine_transition_differential.py`, pinned by
`tests/test_final_holdout_guard.py` -- which refuses the whole of `19,200,000`+
without an explicit opt-in. That is the enforcement half; this is the record-keeping
half. Do not read a green run here as evidence that a band has been measured once.

AND NOTHING HERE IS SKIPPED. Every JSON in the corpus must parse; see `_load_or_die`.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

LEDGER = REPO / "docs" / "engine_divergence_ledger_20260728.md"

# Below this, seeds belong to the acceptance namespaces and to the pre-C28 local
# corpora (`600,000`, `1,350,000`, `1,500,000`, `17,000,000`). Those are governed by
# `PUBLIC_CONSUMED_SEED_RANGES` in `tests/test_cert_contract_registration.py`, which
# is the enforced registry for them. This module covers the fidelity-differential
# namespace only, which the ledger opens at `19,000,000`.
FIDELITY_SEED_FLOOR = 19_000_000

# The registered final-holdout block, and the 60 seeds of it that C141 did not reach.
FINAL_HOLDOUT_REGISTERED = (19_200_000, 19_200_199)
FINAL_HOLDOUT_UNSWEPT_HEAD = (19_200_000, 19_200_059)

# Every band of fidelity seed space that the committed record actually touches.
# MEASURED by running `_seed_intervals` over all 374 committed JSON under `reports/`
# and `docs/` and taking the union of what came back, NOT transcribed from the
# ledger -- the ledger is the thing under test.
#
# Adding a band here is only correct after the sweep that fills it is committed.
# Widening one to make a red test green is the failure mode this pin exists to
# catch: it means a sweep ran outside its registration, which is a fact about the
# record, not about the test.
REGISTERED_BANDS: tuple[tuple[int, int, str], ...] = (
    (19_000_000, 19_000_199, "dev window, swept continuously"),
    (19_100_000, 19_100_199, "validation holdout, C116 Phase 1"),
    (19_200_060, 19_200_259, "final holdout, the span C141 actually swept"),
    (19_500_000, 19_500_799, "c73's 800-game sweep"),
)

# The two witnesses that a `reports/artifacts/` + `seeds.min` selector cannot see or
# would misread. Named individually because the count-based anti-vacuity check below
# stays green when exactly these vanish, and they are the two that matter.
C73 = "reports/c73_eight_hundred_game_sweep.json"
C141_SWEEP = "reports/artifacts/c141_final_holdout_sweep.json"

# Anti-vacuity floor for the corpus walk, not an exact count: every fix branch adds
# dev/holdout sweep pairs, so an exact figure here would be stale within the week and
# would say nothing the band equality below does not already say. 93 at `553cf2c3`.
_MIN_FIDELITY_ARTIFACTS = 80

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

_CORPUS_TREES = ("reports", "docs")


def _seed_intervals(document: object) -> list[tuple[int, int]]:
    """Every closed seed interval this document evidences, at any depth.

    Shape-agnostic by construction -- see the module docstring for why, and for the
    four shapes the real corpus turned out to use.
    """

    found: list[tuple[int, int]] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            low, high = node.get("min"), node.get("max")
            if isinstance(low, int) and isinstance(high, int) and not isinstance(low, bool):
                found.append((low, high))
            for key, value in node.items():
                lowered = key.lower()
                if isinstance(value, int) and not isinstance(value, bool) and "seed" in lowered:
                    if "start" in lowered:
                        end = node.get("seed_end")
                        games = node.get("games")
                        if isinstance(end, int) and not isinstance(end, bool):
                            found.append((value, end))
                        elif isinstance(games, int) and not isinstance(games, bool) and games > 0:
                            found.append((value, value + games - 1))
                        else:
                            found.append((value, value))
                    else:
                        found.append((value, value))
                elif isinstance(value, list) and "seed" in lowered:
                    for element in value:
                        if isinstance(element, int) and not isinstance(element, bool):
                            found.append((element, element))
                if isinstance(value, (dict, list)):
                    visit(value)
        elif isinstance(node, list):
            for element in node:
                if isinstance(element, (dict, list)):
                    visit(element)

    visit(document)
    return found


def _json_files(root: Path, trees: tuple[str, ...] = _CORPUS_TREES) -> list[Path]:
    files: list[Path] = []
    for tree in trees:
        base = root / tree
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in sorted(dirnames) if d not in _SKIP_DIRS]
            for name in sorted(filenames):
                if name.endswith(".json"):
                    files.append(Path(dirpath) / name)
    return sorted(files)


def _load_or_die(paths: list[Path], root: Path, corpus: str) -> list[tuple[Path, object]]:
    """Parse every path, or raise naming all of them. **Nothing is skipped.**

    An earlier revision of this module swallowed `JSONDecodeError`/`UnicodeDecodeError`
    and continued. `tests/test_never_fired_counter_census.py` removed exactly that
    handler for exactly this reason, and reintroducing it here was a real regression
    rather than a stylistic one: every assertion in this module is of the form "no
    committed seed escapes its band", so a file that stops parsing makes the claim
    EASIER and the suite greener. The corpus floor does not save it -- 93 files reach
    fidelity space against `_MIN_FIDELITY_ARTIFACTS`, so up to 13 could go unreadable
    with the count pin still satisfied.

    A file that cannot be read is a red gate, not one fewer haystack.
    """

    loaded: list[tuple[Path, object]] = []
    unreadable: list[str] = []
    for path in paths:
        try:
            loaded.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            unreadable.append(f"{path.relative_to(root).as_posix()}: {type(exc).__name__}: {exc}")
    if unreadable:
        raise AssertionError(
            f"committed JSON in the {corpus} could not be read, so the evidence base "
            "for every seed-coverage assertion in this module is incomplete and every "
            "one of them got easier. Fix or remove the file; do not let it drop out "
            "silently: " + "; ".join(unreadable)
        )
    return loaded


def fidelity_intervals(root: Path = REPO) -> dict[str, list[tuple[int, int]]]:
    """`repo-relative path -> intervals reaching into fidelity seed space`."""

    out: dict[str, list[tuple[int, int]]] = {}
    for path, document in _load_or_die(_json_files(root), root, "reports/ and docs/ corpus"):
        reaching = sorted(
            {
                (low, high)
                for low, high in _seed_intervals(document)
                if high >= FIDELITY_SEED_FLOOR and low <= high
            }
        )
        if reaching:
            out[path.relative_to(root).as_posix()] = reaching
    return out


def unregistered(
    intervals: dict[str, list[tuple[int, int]]],
    bands: tuple[tuple[int, int, str], ...] = REGISTERED_BANDS,
) -> list[tuple[str, int, int]]:
    """Intervals not wholly inside one registered band. Empty means the record agrees."""

    escapes: list[tuple[str, int, int]] = []
    for path, spans in sorted(intervals.items()):
        for low, high in spans:
            if not any(start <= low and high <= end for start, end, _ in bands):
                escapes.append((path, low, high))
    return escapes


def witnesses(
    intervals: dict[str, list[tuple[int, int]]],
    bands: tuple[tuple[int, int, str], ...] = REGISTERED_BANDS,
) -> dict[tuple[int, int], set[str]]:
    """`band -> the committed files that evidence a sweep inside it`."""

    out: dict[tuple[int, int], set[str]] = {(s, e): set() for s, e, _ in bands}
    for path, spans in intervals.items():
        for low, high in spans:
            for start, end, _ in bands:
                if start <= low and high <= end:
                    out[(start, end)].add(path)
    return out


def naive_artifacts_seeds_min(root: Path = REPO) -> dict[str, tuple[int, int]]:
    """The selector that produced the false "virgin above 19,200,260" claim.

    Reconstructed verbatim rather than described, so the miss it causes is a
    measurement in this suite and not a story told about one.
    """

    out: dict[str, tuple[int, int]] = {}
    base = root / "reports" / "artifacts"
    # Same no-swallow rule as `fidelity_intervals`, and needed for the same reason:
    # this selector's ceiling of exactly `19,200,259` is asserted below, and a member
    # dropping out silently could only lower it.
    for path, document in _load_or_die(sorted(base.glob("*.json")), root, "naive selector's corpus"):
        if not isinstance(document, dict):
            continue
        seeds = document.get("seeds")
        if isinstance(seeds, dict) and isinstance(seeds.get("min"), int):
            low = seeds["min"]
            high = seeds.get("max")
            out[path.relative_to(root).as_posix()] = (low, high if isinstance(high, int) else low)
    return out


_TABLE_HEADER = "| namespace | range | purpose | status |"
_SECTION_END = "**The invariant, restated against the ACTIVE registration.**"


def registry_table(text: str | None = None) -> str:
    """Just the seed-registry table's rows.

    Scoped deliberately. Asserting `assertNotIn("reserved, untouched", <whole file>)`
    is wrong twice over: the amendment note below the table QUOTES the false status it
    is retracting, so the pin would be unsatisfiable, and a failure would dump 380 KB
    of ledger into the log. A status claim belongs to the table; assert it there.
    """

    body = _ledger_text() if text is None else text
    start = body.index(_TABLE_HEADER)
    rows = []
    for line in body[start:].splitlines():
        if not line.startswith("|"):
            break
        rows.append(line)
    return "\n".join(rows)


def registry_section(text: str | None = None) -> str:
    """The table plus the prose amendments that qualify it, up to the next invariant."""

    body = _ledger_text() if text is None else text
    start = body.index(_TABLE_HEADER)
    end = body.index(_SECTION_END, start)
    return body[start:end]


def _ledger_text() -> str:
    return LEDGER.read_text()


class FidelitySeedCoverageTests(unittest.TestCase):
    """The registry, re-derived from the artifacts rather than read."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.intervals = fidelity_intervals()

    def test_the_corpus_is_not_empty(self) -> None:
        # Anti-vacuity. Every assertion below is over `self.intervals`; a selector
        # that returns nothing makes all of them pass, which is precisely the
        # fail-open shape that let the false row stand.
        self.assertGreaterEqual(
            len(self.intervals), _MIN_FIDELITY_ARTIFACTS,
            "the fidelity-space corpus collapsed; the selector is broken, not the record",
        )

    def test_the_two_witnesses_a_naive_selector_would_lose_are_present(self) -> None:
        # A count floor stays green when exactly these two go missing, and these two
        # are the entire subject of this module.
        self.assertIn(C73, self.intervals)
        self.assertIn(C141_SWEEP, self.intervals)

    def test_every_committed_fidelity_seed_lies_in_a_registered_band(self) -> None:
        escapes = unregistered(self.intervals)
        self.assertEqual(
            escapes, [],
            "a committed artifact reaches fidelity seeds outside every registered band. "
            "Either a sweep ran outside its registration, or REGISTERED_BANDS is stale. "
            "Do not widen the band to make this green without recording which sweep "
            "produced the seeds.",
        )

    def test_every_registered_band_has_a_committed_witness(self) -> None:
        # The other direction. Without this, deleting `c73` or C141's sweep artifact
        # leaves the module green while the registry keeps claiming both bands are
        # consumed -- an absence-only pin's fail-open.
        empty = [band for band, files in witnesses(self.intervals).items() if not files]
        self.assertEqual(
            empty, [],
            "a band the registry records as swept has no committed artifact evidencing it",
        )

    def test_the_final_holdout_band_is_the_span_the_artifact_reports(self) -> None:
        # Read the artifact at test time. The band constant and the ledger row must
        # both answer to it, so the numbers cannot drift apart in either direction.
        sweep = json.loads((REPO / C141_SWEEP).read_text())
        self.assertEqual(sweep["seeds"]["min"], 19_200_060)
        self.assertEqual(sweep["seeds"]["max"], 19_200_259)
        self.assertEqual(sweep["seeds"]["distinct"], 200)
        self.assertEqual(sweep["games"], 200)
        self.assertIn(
            (sweep["seeds"]["min"], sweep["seeds"]["max"], "final holdout, the span C141 actually swept"),
            REGISTERED_BANDS,
        )

    def test_the_swept_span_overruns_the_registered_block(self) -> None:
        # Stated as an arithmetic identity so the "60 seeds" in the ledger row is
        # derived here rather than asserted there.
        swept_start, swept_end, _ = REGISTERED_BANDS[2]
        registered_start, registered_end = FINAL_HOLDOUT_REGISTERED
        self.assertEqual(swept_end - registered_end, 60)
        self.assertEqual(swept_start - registered_start, 60)
        self.assertEqual(FINAL_HOLDOUT_UNSWEPT_HEAD, (registered_start, swept_start - 1))

    def test_the_unswept_head_has_no_committed_artifact(self) -> None:
        # `19,200,000`-`19,200,059`. The 60 games that ran here before the guard
        # existed were deleted unread, so the absence of an artifact is the whole of
        # what the record can say about them -- and it is what the corrected row says.
        low, high = FINAL_HOLDOUT_UNSWEPT_HEAD
        touching = {
            path: spans
            for path, spans in self.intervals.items()
            for start, end in spans
            if start <= high and end >= low
        }
        self.assertEqual(
            touching, {},
            "something in the committed record now covers the unswept head of the "
            "final holdout; the ledger row saying it is unswept is no longer true",
        )


class TheNaiveSelectorIsTheRecordedMistakeTests(unittest.TestCase):
    """Pin the miss that cost the time, against the real tree.

    Not a synthetic demonstration: the numbers below are the ones that actually
    produced the false claim.
    """

    def test_the_naive_selector_cannot_see_c73(self) -> None:
        naive = naive_artifacts_seeds_min()
        self.assertGreaterEqual(len(naive), 60, "anti-vacuity: the naive selector found nothing")
        self.assertNotIn(C73, naive)
        self.assertIn(C141_SWEEP, naive)

    def test_the_naive_selector_tops_out_at_the_final_holdout(self) -> None:
        # This exact number is why "everything at or above 19,200,260 is virgin" felt
        # safe to assert. It is the naive selector's ceiling, not the record's.
        naive = naive_artifacts_seeds_min()
        self.assertEqual(max(high for _, high in naive.values()), 19_200_259)

    def test_the_shape_agnostic_selector_does_see_c73(self) -> None:
        intervals = fidelity_intervals()
        self.assertIn(C73, intervals, "the shape-agnostic selector lost c73 too")
        spans = intervals[C73]
        self.assertIn((19_500_000, 19_500_799), spans)
        self.assertGreater(19_500_799, 19_200_259)


class TheSelectorItselfIsExercisedTests(unittest.TestCase):
    """Prove the pin fires. Four inert pins have shipped in this repo already.

    Each case drives `_seed_intervals` / `unregistered` / `witnesses` with input that
    MUST be rejected, so a regression that turns the selector into a no-op fails here
    rather than passing quietly over the real corpus.
    """

    def test_the_seeds_min_max_shape_is_read(self) -> None:
        self.assertIn(
            (19_000_000, 19_000_199),
            _seed_intervals({"seeds": {"min": 19_000_000, "max": 19_000_199}}),
        )

    def test_the_run_seed_start_shape_is_read_and_closed_by_games(self) -> None:
        # `c73`'s shape. Closing the interval with `games` is what makes the 800-game
        # sweep's true end -- `19,500,799` -- visible at all; the JSON never states it.
        self.assertIn(
            (19_500_000, 19_500_799),
            _seed_intervals({"run": {"seed_start": 19_500_000, "games": 800}}),
        )

    def test_the_sample_shape_is_read_and_seed_end_wins_over_games(self) -> None:
        # `c82`'s shape, which carries both. An explicit end is evidence; a count is
        # an inference from it, so the explicit one is preferred.
        self.assertIn(
            (19_000_000, 19_000_199),
            _seed_intervals(
                {"sample": {"seed_start": 19_000_000, "seed_end": 19_000_199, "games": 200}}
            ),
        )

    def test_the_nested_windows_shape_is_read(self) -> None:
        # `c147_g33b_gate_reach`'s shape: two spans, neither at the top level.
        spans = _seed_intervals(
            {
                "windows": {
                    "dev": {"seed_start": 19_000_000, "games": 200},
                    "holdout": {"seed_start": 19_100_000, "games": 200},
                }
            }
        )
        self.assertIn((19_000_000, 19_000_199), spans)
        self.assertIn((19_100_000, 19_100_199), spans)

    def test_a_bare_repro_seed_is_read(self) -> None:
        self.assertIn(
            (19_200_075, 19_200_075),
            _seed_intervals({"repros": [{"seed": 19_200_075, "step": 12}]}),
        )

    def test_a_seed_in_the_unswept_head_is_rejected(self) -> None:
        # The mutation that matters most: any future artifact covering
        # `19,200,000`-`19,200,059` must turn this module red.
        escapes = unregistered({"synthetic.json": [(19_200_000, 19_200_059)]})
        self.assertEqual(escapes, [("synthetic.json", 19_200_000, 19_200_059)])

    def test_a_sweep_above_the_top_band_is_rejected(self) -> None:
        escapes = unregistered({"synthetic.json": [(19_600_000, 19_600_799)]})
        self.assertEqual(escapes, [("synthetic.json", 19_600_000, 19_600_799)])

    def test_a_span_straddling_a_band_edge_is_rejected(self) -> None:
        # Containment, not intersection. A sweep that starts inside a registered band
        # and runs past its end is the exact defect C141 committed, and a rule keyed
        # on "overlaps a band" would have passed it.
        escapes = unregistered({"synthetic.json": [(19_200_060, 19_200_260)]})
        self.assertEqual(escapes, [("synthetic.json", 19_200_060, 19_200_260)])

    def test_a_band_with_no_witness_is_reported(self) -> None:
        bands = REGISTERED_BANDS + ((19_900_000, 19_900_099, "synthetic, unwitnessed"),)
        found = witnesses({"synthetic.json": [(19_000_000, 19_000_199)]}, bands)
        self.assertEqual(found[(19_900_000, 19_900_099)], set())
        self.assertEqual(found[(19_000_000, 19_000_199)], {"synthetic.json"})

    def test_an_unparseable_artifact_is_a_red_gate_not_one_fewer_haystack(self) -> None:
        # The regression this module shipped once: swallowing a decode error made every
        # "no seed escapes its band" claim easier while keeping the suite green, and the
        # `_MIN_FIDELITY_ARTIFACTS` floor has 13 files of slack, so it cannot notice.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reports" / "artifacts").mkdir(parents=True)
            (root / "reports" / "good.json").write_text(
                json.dumps({"seeds": {"min": 19_000_000, "max": 19_000_199}}), encoding="utf-8"
            )
            (root / "reports" / "truncated.json").write_text('{"seeds": {"min": 1', encoding="utf-8")
            with self.assertRaises(AssertionError) as caught:
                fidelity_intervals(root)
        self.assertIn("truncated.json", str(caught.exception))
        self.assertIn("JSONDecodeError", str(caught.exception))

    def test_the_naive_selector_refuses_an_unparseable_artifact_too(self) -> None:
        # Its ceiling of exactly 19,200,259 is asserted as an equality above. A member
        # dropping out silently could only move that number, so it gets the same rule.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reports" / "artifacts").mkdir(parents=True)
            (root / "reports" / "artifacts" / "bad.json").write_bytes(b"\xff\xfe not json")
            with self.assertRaises(AssertionError) as caught:
                naive_artifacts_seeds_min(root)
        self.assertIn("bad.json", str(caught.exception))

    def test_a_readable_synthetic_tree_still_parses(self) -> None:
        # Anti-vacuity for the two pins above: they would also pass if `_load_or_die`
        # raised unconditionally, which would make the whole module a false alarm.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reports").mkdir(parents=True)
            (root / "reports" / "good.json").write_text(
                json.dumps({"run": {"seed_start": 19_500_000, "games": 800}}), encoding="utf-8"
            )
            self.assertEqual(
                fidelity_intervals(root), {"reports/good.json": [(19_500_000, 19_500_799)]}
            )

    def test_seeds_below_the_fidelity_floor_are_out_of_scope(self) -> None:
        # `1,500,000` and `17,000,000` are governed by PUBLIC_CONSUMED_SEED_RANGES.
        # Pulling them in here would make this module fight that one.
        self.assertNotIn("reports/c9_summary.json", fidelity_intervals())


class TheLedgerRowSaysWhatTheArtifactsSayTests(unittest.TestCase):
    """The prose side, tied to numbers this module derives rather than to a transcript."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.table = registry_table()
        cls.section = registry_section()

    @staticmethod
    def _grouped(value: int) -> str:
        return f"{value:,}"

    def test_the_table_and_section_are_non_empty(self) -> None:
        # Anti-vacuity for every pin below. `assertNotIn` against an accidentally
        # empty slice passes, and that is the whole class of failure this module was
        # written to stop.
        self.assertGreaterEqual(len(self.table.splitlines()), 10)
        # Six fidelity rows: dev, c73, validation holdout, and the three the final
        # holdout now decomposes into. Exact, so collapsing the decomposition back to
        # one row -- which is how the false status got there -- is red.
        self.assertEqual(self.table.count("fidelity differential"), 6)
        self.assertGreater(len(self.section), 2_000)

    def test_the_false_status_is_gone_from_the_table(self) -> None:
        # The literal string the row carried at `553cf2c3`. It asserted that a band
        # with a committed 200-game sweep had never been touched. It survives in the
        # prose below the table, as a quoted retraction, which is why this is scoped.
        self.assertNotIn("reserved, untouched", self.table)
        self.assertIn("reserved, untouched", self.section)

    def test_the_table_records_every_band_this_module_derives(self) -> None:
        for start, end, label in REGISTERED_BANDS:
            self.assertIn(self._grouped(start), self.table, f"{label}: start missing")
            self.assertIn(self._grouped(end), self.table, f"{label}: end missing")

    def test_the_table_records_the_unswept_head_and_the_registered_block(self) -> None:
        for value in (*FINAL_HOLDOUT_REGISTERED, *FINAL_HOLDOUT_UNSWEPT_HEAD):
            self.assertIn(self._grouped(value), self.table)

    def test_the_table_no_longer_records_an_open_ended_c73_band(self) -> None:
        # `19,500,000+` was true but unusable: it gives a scanner no end to check
        # against, and it silently swallows the final-holdout block, which is why the
        # "exactly one measurement above 19,200,000" invariant read as satisfied.
        self.assertNotIn("`19,500,000`+", self.table)

    def test_the_section_names_all_four_artifact_shapes(self) -> None:
        # The recorded lesson. If this note is deleted, the next scan narrows back to
        # `seeds.min` under `reports/artifacts/` and misses `c73` again.
        for shape in ("seeds.min", "run.seed_start", "sample.seed_start", "windows."):
            self.assertIn(shape, self.section, f"the {shape} shape is no longer named")
        self.assertIn("c73_eight_hundred_game_sweep.json", self.section)

    def test_the_section_points_at_the_enforced_version(self) -> None:
        self.assertIn("tests/test_seed_registry_coverage.py", self.section)


if __name__ == "__main__":
    unittest.main()
