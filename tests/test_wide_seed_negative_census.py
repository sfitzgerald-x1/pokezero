"""Re-derive C153's wide-seed verdict partition from the committed census artifacts.

WHY THIS MODULE EXISTS, AND WHY IT IS NOT `tests/test_never_fired_counter_census.py`.

#1200 added a standing rule to `reports/c138_known_gaps_ledger.md` §8:

  > A negative measured only inside the two permitted windows is a claim about those
  > windows. Widening the CORPUS cannot find this class of error; only widening the
  > MEASUREMENT can.

`tests/test_never_fired_counter_census.py` is the corpus instrument. It re-derives every
absence over 401 committed JSON on every run, and the rule above says in its own sentence
that this cannot catch the class of error it is about -- C152's two refutations are 0 in
every artifact committed before C152 and fire immediately on unregistered seeds. The rule
therefore landed with **no instrument behind it**, and its reviewer said so. This module
is that instrument: it pins a MEASUREMENT taken somewhere new, not a scan taken somewhere
wider.

A separate module rather than more classes in the census, for the reason C150 recorded:
the census's workflow step carries an exact `Ran 22 tests` guard, and growing it silently
weakens a guard that exists to notice growth.

WHAT IS PINNED.

  1. **The inventory is DERIVED and it is EXACTLY covered.** `scripts/c153_wide_negative_census.py`
     builds the inventory from source by AST -- the 40 `EngineWorldUnsupported` reasons,
     the 19 `classify_divergence` classes, the 8 `UnmappableChoice` reasons -- and this
     module re-runs that derivation and asserts the committed artifact's verdict map has
     exactly those keys, in both directions. A reason added to `engine_world.py` therefore
     joins the inventory and reddens this module until it is measured, which is the
     failure mode §3.5 has had seven times.

  2. **Every entry has an emittable counter key.** An inventory row whose key the harness
     cannot emit is not a negative that survived a census; it is a check that asserts
     nothing, and this repo has shipped eight of those. Derived from the differential's
     `counts[...]` key space by AST.

  3. **The fired set is exact SET EQUALITY, both directions**, exactly as the corpus census
     does it. A counter that starts firing is red; so is one that stops.

  4. **Every firing re-derives from the committed shards**, not from the census artifact's
     own summary -- the artifact is re-scanned against the shard counters it claims to
     summarise, so a hand-edited census JSON cannot make a claim the shards do not carry.

  5. **The per-seed attribution CLOSES against the shard totals.** The seeds come from
     per-game checkpoints that are deliberately not committed (10,000 records). That is
     only acceptable because the sum of the per-seed values must equal the committed shard
     counter, so a missing or invented seed breaks the closure rather than leaving an
     unfalsifiable sentence.

  6. **The scope is outside every reserved band**, asserted against spans read from the
     artifacts rather than against numbers retyped here.

  7. **The build is the SHIPPING build.** C152's wide census ran on the instrumented
     throwaway `89797289...` with harness digest `e3459e1f...`; this one is
     `bfdbe1c04876edcd` at the shipped harness. Pinned so the two are never pooled by a
     later reader, and so a re-run on a mutant build cannot inherit these verdicts.

  8. **Anti-vacuity.** Four counters that DO fire in the census are asserted nonzero. Every
     absence pin here is a loop over the same twelve shards, and a loop over a shard set
     that stopped being read passes.

THE MUTATION BATTERY, ENUMERATED. The sibling modules state theirs in the docstring
(`tests/test_ledger_table_uniformity.py`, "9 mutations applied, 9 caught, listed in the
module docstring"), and this one did not -- it claimed "12 applied, 12 caught" and named
none. An independent reviewer then re-ran a battery of 13 and found the 13th survives.
**That is exactly what an unrecorded battery costs**, so the list is here and a mutation
added to it must be added here too. 14-18 came with the §6 sync pin, added after a second
review round found that section still carrying pre-correction prose; 19-22 came with the
derived emission-granularity split, after a third found that split asserted and untraced;
23-25 are review's own smuggles against the negative pin's first exemption rule, which they
defeated -- all three are green under the window version and red under the quoting one.

Each is applied to a copy of the committed artifact (or the generator) and must turn this
module RED:

   1. a `FIRED` verdict flipped to `NOT_OBSERVED_AT_SCOPE`
   2. an inventory entry dropped from the artifact's verdict map
   3. a per-seed total perturbed AND the `agrees` flag forced back to true
   4. a shard span rewritten into the dev window
   5. a control counter zeroed
   6. the engine fingerprint swapped for C152's instrumented `89797289...`
   7. a scope sentence replaced by a bare "never fired"
   8. `CENSUS_CANNOT_REACH` emptied in the generator
   9. a witness seed replaced by one outside every shard
  10. C152's refutations zeroed on the strict arm
  11. an entry with no emittable counter key admitted
  12. the strict per-divergence bound detached from the shards
  13. ⚠ **the COMBINED per-divergence bound detached from the shards** -- set to
      `3/803264`, the per-boundary substitution this module's own comment calls a
      four-orders-of-magnitude overstatement
  14. §6's headline count changed from Six to Seven
  15. §6's structural subsection count changed from (2) to (3)
  16. §6's input-group subsection count changed from (4) to (5)
  17. a §6 bullet de-listed so the section names one fewer entry than the generator
  18. `public_effect_blocked` re-listed among §6's unreachable bullets
  19. an entry moved between granularity buckets in `emission_granularity`
  20. a recorded `denominator_trials` value detached from the shards
  21. an entry's `denominator` name swapped for another of the three
  22. the derived `liveness_witnesses` list edited away from what the AST produces
  23. a retracted phrase smuggled UNQUOTED into a §6 bullet (which opens with `⚠`)
  24. the same, capitalised, to defeat a case-sensitive match
  25. a bare `⚠ Housekeeping note.` two lines above an unquoted assertion

⚠ 13 was found by the reviewer, not the author, and it was the important one: `combined`
is where the 0.32 % bound quoted in the report, in §3.5 and in H15's cell comes from, and
the first revision of `test_the_stated_bounds_...` looped over `document["arms"]`, which
holds only `strict` and `banded`. Four separate mutations of the combined block passed
green. Three of the twelve above (3, 8 and 13's neighbours) were likewise near-misses at
first: 3 survived until the closure pin stopped reading the artifact's own `agrees` flag,
and 8's first form was a defective mutation (`{} or {...}`, which is truthy) rather than a
surviving pin. Both are recorded rather than tidied away.

CROSS-INSTRUMENT COUPLING, DECLARED. This module and
`tests/test_never_fired_counter_census.py` check each other, and until now that was true
by arrangement rather than by statement:

  * THIS module pins a MEASUREMENT taken outside the two permitted windows; that one pins
    an ABSENCE over the committed corpus. Neither can do the other's job, which is the
    whole content of §8's standing rule.
  * The corpus module is what actually enforces "no number may sit under a dotted path
    containing a counter name" in the artifact this module owns. It caught that defect on
    the same tree, in three variants, by shape. The two `assertNotIn`s here are a
    regression pin for the instance, not the invariant.
  * ⚠ RESIDUAL RISK: that enforcement depends on
    `reports/artifacts/c153_wide_negative_census.json` being a member of
    `counter_artifacts()`'s glob. Moving a future census to `tests/data/` would drop it out
    of the corpus and lose the check with no test going red. Asserted here as a floor, and
    named in both modules so the dependency is deliberate.

WHAT IS NOT PINNED, ON PURPOSE. Nothing here asserts a divergence rate. These are
unregistered seeds and the census is not fidelity evidence; §7.3 of
`reports/c152_ledger_terminal_disposition.md` says so about its own predecessor and the
same applies with more force to a sample twenty-five times the size of the two windows.
"""

from __future__ import annotations

import importlib.util
import json
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CENSUS = REPO / "reports/artifacts/c153_wide_negative_census.json"


def _load_census_script():
    """Import the generator, so the inventory is DERIVED here and not transcribed.

    Imported by path rather than by adding `scripts/` to `sys.path` permanently: that
    directory holds several modules whose names collide with installed packages, and a
    test that mutates import state for the whole run is a source of order-dependent green.
    """

    spec = importlib.util.spec_from_file_location(
        "_c153_wide_negative_census", REPO / "scripts/c153_wide_negative_census.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CENSUS_SCRIPT = _load_census_script()


def _document() -> dict:
    return json.loads(CENSUS.read_text(encoding="utf-8"))


# THE SHIPPING BUILD this census was taken on. Not C152's `89797289...` instrumented
# throwaway, and the difference is load-bearing: pooling the two would quote a 9,000-game
# figure taken on two different engines.
_EXPECTED_ENGINE_FINGERPRINT = (
    "bfdbe1c04876edcd1957e7a360c5086cfc7eae32ccf3ba0e71d137bd76df3990"
)
_C152_INSTRUMENTED_FINGERPRINT_PREFIX = "89797289"

# Bands no census may touch. `19,200,000`-`19,200,259` is burned; `19,300,000`-`19,300,199`
# is the owner-ratified one-shot window; the two 200-seed windows are the permitted ones,
# and a census inside them would be the very thing this module exists to escape.
_FORBIDDEN_BANDS = (
    (19_000_000, 19_000_199, "dev window"),
    (19_100_000, 19_100_199, "validation holdout"),
    (19_200_000, 19_200_259, "BURNED final-holdout block"),
    (19_300_000, 19_300_199, "OWNER_RATIFIED replacement window"),
)

# Below this the seed registry does not track a band at all, which is why these seeds need
# no registration and can never be confused with fidelity evidence. Kept as a literal
# rather than imported from `tests/test_seed_registry_coverage.py`: importing a sibling
# test module by name only works under some invocations, and a pin that silently changes
# shape between `python -m unittest` and `python <file>` is the defect #1112 and #1200 both
# shipped. The value is asserted against that module's constant by its own pin.
_FIDELITY_SEED_FLOOR = 19_000_000

# ⚠ THE RESULT. Set equality in both directions. Every member is a row-level negative this
# program asserted at a scope it had never measured at, and every one of them was found by
# MEASURING somewhere new rather than by scanning more artifacts.
_FIRED_AT_WIDE_SCOPE: frozenset[str] = frozenset(
    {
        # H15's four "reachable only through `--matcher banded`, which no committed
        # artifact used". THREE of the four are now measured on that path, which
        # discharges the scope caveat rather than refuting the claim.
        "divergence_class:damage_band",  # 375, over 313 distinct seeds
        "divergence_class:faint_boundary",  # 30, over 29 distinct seeds
        "divergence_class:status_support",  # 84, over 71 distinct seeds
        # ⚠ H15's six "strict-path classes the program has simply never produced".
        # FOUR of the six fire, and the cell's CATEGORY is what is wrong: these are
        # protocol-evidence fallbacks reached exactly when the miss reason could not be
        # parsed -- `classify_divergence`'s own comment marks the whole tail "Banded
        # matcher (or an unparsable miss): fall back to protocol evidence" -- so they
        # belong with the three above and not with `component_set_equal_but_unmatched`,
        # which really is on the strict component path.
        "divergence_class:evidence:crit_in_step",  # 3
        "divergence_class:evidence:faint_ply_no_upkeep",  # 30
        "divergence_class:evidence:spikes_in_step",  # 2
        "divergence_class:unclassified",  # 163 -- 23.7 % of the banded arm's divergences
        #
        # NOT HERE, and that is the headline: every one of §3.5's 46 window-scoped
        # verified negatives is still absent. 8,000 strict games and 641,866 measured
        # boundaries did not move one of them. See
        # `reports/c153_wide_seed_negative_census.md` for what that does and does not
        # license.
    }
)


class TheInventoryIsDerivedAndExactlyCoveredTests(unittest.TestCase):
    def test_the_verdict_map_is_exactly_the_derived_inventory(self) -> None:
        derived = set(CENSUS_SCRIPT.inventory())
        recorded = set(_document()["verdicts"])
        self.assertEqual(
            recorded,
            derived,
            "the committed census no longer covers the inventory its own generator "
            "derives.\n"
            f"  missing a verdict: {sorted(derived - recorded)}\n"
            f"  verdict with no inventory entry: {sorted(recorded - derived)}\n"
            "A refusal reason or divergence class added to source joins the inventory "
            "automatically; re-run scripts/c153_wide_negative_census.py and give it a "
            "verdict rather than widening this pin.",
        )

    def test_the_taxonomy_sizes_match_the_source_they_were_derived_from(self) -> None:
        self.assertEqual(
            _document()["taxonomy_sizes"],
            {
                "world_unsupported_reasons": len(CENSUS_SCRIPT.world_unsupported_reasons()),
                "divergence_classes": len(CENSUS_SCRIPT.divergence_classes()),
                "unmappable_choice_reasons": len(CENSUS_SCRIPT.unmappable_choice_reasons()),
            },
            "the committed census was taken against a different taxonomy than source "
            "carries today",
        )

    def test_section_3_5s_inventory_is_fifty_and_forty_six_are_window_scoped(self) -> None:
        # RE-DERIVED, and it corrected the brief this work started from. §3.5's four
        # verified-negative lists are 8 + 6 + 7 + 29 = 50. FOUR of the 50 carry an
        # argument independent of measurement -- `mapper_lossy` and `no_usable_branch`
        # (structural), `nature_not_neutral` (R7) and `weather_unsupported` (R8) -- so
        # 46, not 45, rest purely on window-scoped measurement. The fifth
        # measurement-independent name usually listed, `future_sight_pending`, is NOT a
        # member of the 50: §3.5 retires it out of the list of 33 BEFORE the four
        # corrections that take 33 to 29. Subtracting it produced the 45.
        sizes = _document()["inventory_size"]
        self.assertEqual(sizes["section_3_5_verified_negatives"], 50)
        self.assertEqual(sizes["window_scoped"], 46)

    def test_every_inventory_entry_has_a_counter_key_the_harness_can_emit(self) -> None:
        self.assertEqual(
            _document()["entries_with_no_emittable_counter_key"],
            [],
            "an inventory entry names a counter key the differential cannot emit, so its "
            "'not observed' verdict measures nothing. That is the check-that-asserts-"
            "nothing shape, not a negative result.",
        )


class TheFiredPartitionAtWideScopeTests(unittest.TestCase):
    def test_the_fired_set_is_exactly_pinned(self) -> None:
        fired = {
            name
            for name, record in _document()["verdicts"].items()
            if record["verdict"] == "FIRED"
        }
        self.assertEqual(
            fired,
            set(_FIRED_AT_WIDE_SCOPE),
            "the wide-scope fired/not-observed partition moved.\n"
            f"  newly fired: {sorted(fired - set(_FIRED_AT_WIDE_SCOPE))}\n"
            f"  no longer firing: {sorted(set(_FIRED_AT_WIDE_SCOPE) - fired)}\n"
            "A newly fired name needs a ledger row in the same commit; a name that "
            "stopped firing means the shards or the scanner changed and the ledger's "
            "correction is now unsupported.",
        )

    def test_every_firing_records_the_arm_it_fired_on(self) -> None:
        # The distinction the whole H15 correction turns on. `--matcher banded` is the
        # legacy net-HP comparator, not the shipping one, so a class that fires only
        # there refutes "never produced" and says nothing about the path that certifies.
        for name, record in sorted(_document()["verdicts"].items()):
            if record["verdict"] != "FIRED":
                continue
            with self.subTest(entry=name):
                self.assertTrue(
                    set(record["arms_that_fired"]) <= {"strict", "banded"},
                    f"{name} records an unknown arm: {record['arms_that_fired']}",
                )
                self.assertTrue(record["arms_that_fired"])

    def test_every_firing_re_derives_from_the_committed_shard_counters(self) -> None:
        """The artifact is re-scanned against the shards, never trusted about them.

        Without this the census JSON is a transcription, and a transcription is what the
        ledger's G8 cell was withdrawn for.
        """

        document = _document()
        shards = [
            (name, json.loads((REPO / name).read_text(encoding="utf-8")))
            for arm in document["arms"].values()
            for name in arm["shards"]
        ]
        entries = CENSUS_SCRIPT.inventory()
        rescanned = CENSUS_SCRIPT.scan_shards(shards)
        for name, record in sorted(document["verdicts"].items()):
            with self.subTest(entry=name):
                if record["verdict"] == "FIRED":
                    self.assertIn(name, rescanned, f"{name} has no shard evidence")
                    self.assertEqual(
                        {k: dict(sorted(v.items())) for k, v in record["evidence"].items()},
                        {k: dict(sorted(v.items())) for k, v in rescanned[name].items()},
                        f"{name}'s recorded evidence disagrees with the shards",
                    )
                else:
                    self.assertNotIn(
                        name,
                        rescanned,
                        f"{name} is recorded as {record['verdict']} but the committed "
                        "shards carry a nonzero counter for it",
                    )
                self.assertIn(name, entries)

    def test_the_per_seed_attribution_closes_against_the_shard_totals(self) -> None:
        """The seeds come from per-game checkpoints that are NOT committed.

        This closure is the whole reason that is acceptable: a seed that was invented, or
        a checkpoint that went missing, has to break a SUM rather than quietly weaken a
        sentence.

        ⚠ RECOMPUTED HERE, NEVER READ. A first revision asserted `record["agrees"]`, the
        artifact's own closure flag, and a mutation battery walked straight through it:
        perturbing a per-seed value and setting `agrees` back to `true` stayed green. A
        field that certifies itself is exactly the defect the ledger's G8 cell was
        withdrawn for. Both sides are rebuilt from primary data below -- the shard side
        from the twelve committed sweeps, the seed side from the per-entry `seeds` map --
        and the recorded flag is checked last, as a transcription rather than as evidence.
        """

        document = _document()
        closure = document["closure"]
        self.assertEqual(
            sorted(closure),
            sorted(_FIRED_AT_WIDE_SCOPE),
            "the closure report and the fired set describe different populations",
        )
        shards = [
            (name, json.loads((REPO / name).read_text(encoding="utf-8")))
            for arm in document["arms"].values()
            for name in arm["shards"]
        ]
        rescanned = CENSUS_SCRIPT.scan_shards(shards)
        for name, record in sorted(closure.items()):
            with self.subTest(entry=name):
                from_shards: dict[str, int] = {}
                for hits in rescanned[name].values():
                    for key, value in hits.items():
                        from_shards[key] = from_shards.get(key, 0) + value
                from_seeds = {
                    key: sum(seeds.values())
                    for key, seeds in document["verdicts"][name]["seeds"].items()
                }
                self.assertEqual(
                    from_seeds,
                    from_shards,
                    f"{name}: the per-seed attribution sums to {from_seeds} against "
                    f"{from_shards} in the committed shards",
                )
                # And the artifact's own transcription of the same two sides.
                self.assertEqual(record["shard_totals"], from_shards)
                self.assertEqual(record["checkpoint_totals"], from_seeds)
                self.assertTrue(record["agrees"])

    def test_every_seed_named_as_a_witness_lies_inside_a_census_shard(self) -> None:
        document = _document()
        spans = [
            (report["seeds"]["min"], report["seeds"]["max"])
            for arm in document["arms"].values()
            for name in arm["shards"]
            for report in [json.loads((REPO / name).read_text(encoding="utf-8"))]
        ]
        witnesses = 0
        for name, record in sorted(document["verdicts"].items()):
            for key, seeds in (record.get("seeds") or {}).items():
                for seed in seeds:
                    witnesses += 1
                    with self.subTest(entry=name, counter=key, seed=seed):
                        self.assertTrue(
                            any(low <= int(seed) <= high for low, high in spans),
                            f"{name} cites seed {seed}, which is in none of the shards",
                        )
        # A loop over nothing passes, and this one would if the seed attribution were
        # dropped from the artifact wholesale.
        self.assertGreater(witnesses, 500, "the per-seed attribution has gone missing")


class TheScopeIsWhatItClaimsToBeTests(unittest.TestCase):
    def test_no_shard_touches_a_reserved_band(self) -> None:
        document = _document()
        for arm, span in document["arms"].items():
            for low, high, label in _FORBIDDEN_BANDS:
                with self.subTest(arm=arm, band=label):
                    self.assertTrue(
                        span["seed_max"] < low or span["seed_min"] > high,
                        f"the {arm} arm ({span['seed_min']}-{span['seed_max']}) overlaps "
                        f"{label} ({low}-{high})",
                    )

    def test_the_seed_floor_of_the_registry_is_not_reached(self) -> None:
        for arm, span in _document()["arms"].items():
            with self.subTest(arm=arm):
                self.assertLess(span["seed_max"], _FIDELITY_SEED_FLOOR)

    def test_the_floor_used_here_is_the_registrys_own(self) -> None:
        # Anti-drift for the literal above: it is retyped, so it is checked against the
        # module that owns it rather than trusted.
        source = (REPO / "tests/test_seed_registry_coverage.py").read_text(encoding="utf-8")
        self.assertIn("FIDELITY_SEED_FLOOR = 19_000_000", source)

    def test_the_census_ran_on_the_shipping_build_and_only_that(self) -> None:
        provenance = _document()["provenance"]
        self.assertTrue(
            provenance["single_build"],
            f"the census spans more than one build: {provenance['distinct']}",
        )
        recorded = json.loads(provenance["distinct"][0])
        self.assertEqual(recorded["engine_fingerprint"], _EXPECTED_ENGINE_FINGERPRINT)
        self.assertFalse(recorded["enumerate_rolls"])
        self.assertNotIn(
            _C152_INSTRUMENTED_FINGERPRINT_PREFIX,
            recorded["engine_fingerprint"],
            "this census must not be taken on C152's instrumented throwaway build; the "
            "two samples are on different engines and must never be pooled",
        )

    def test_the_scope_sentence_is_on_every_verdict(self) -> None:
        # ⚠ The point of the whole exercise. "still not observed" with no scope is the
        # sentence this program has been wrong with seven times, so a verdict without one
        # is a red gate rather than a terse row.
        for name, record in sorted(_document()["verdicts"].items()):
            with self.subTest(entry=name):
                self.assertRegex(
                    record["scope"],
                    r"^[\d,]+ games on unregistered seeds [\d,]+-[\d,]+ "
                    r"\([\d,]+ strict \+ [\d,]+ banded\), [\d,]+ measured boundaries, "
                    r"engine fingerprint [0-9a-f]{16}$",
                    f"{name} carries no stated scope",
                )

    def test_the_census_is_materially_wider_than_the_two_permitted_windows(self) -> None:
        """The claim is "somewhere new AND bigger", so both halves are measured.

        §1.4 of the ledger names the blind spot this addresses in so many words: "a shape
        with a 1-in-50,000 boundary incidence is reachable and would show zero rows" in
        two 200-game windows. The ratio below is what closes that sentence.
        """

        document = _document()
        windows = sum(
            json.loads(
                (REPO / f"reports/artifacts/c152_head_{name}_sweep.json").read_text(
                    encoding="utf-8"
                )
            )["boundaries_measured"]
            for name in ("dev", "holdout")
        )
        census = sum(arm["boundaries_measured"] for arm in document["arms"].values())
        self.assertGreater(
            census,
            15 * windows,
            f"the census measured {census} boundaries against {windows} in the two "
            "permitted windows; a sample that is not materially wider settles nothing "
            "the windows had not already settled",
        )

    def test_the_stated_bounds_are_the_rule_of_three_on_the_measured_sample(self) -> None:
        """Every bound, on every denominator, recomputed from the twelve committed shards.

        ⚠ REWRITTEN 2026-08-08 after review. The first revision looped
        `for arm, span in document["arms"].items()`, and `arms` holds only `strict` and
        `banded` -- so `combined` got an `assertIn` and nothing else. **The single most
        load-bearing number in this work was the one number no pin recomputed:** the
        0.32 % per-divergence bound quoted in the report, in §3.5 and in H15's cell comes
        out of `combined`. An independent battery walked four mutations through it,
        including setting combined `rule_of_three_per_divergence_upper_95` to `3/803264`
        -- which is literally the 1-in-267,755 substitution the comment below says
        overstates the census by four orders of magnitude. A pin that names a trap and
        then leaves it open is worse than no pin, because the comment reads as coverage.

        Two changes, not one. `combined` is now a span like the others, and every span is
        rebuilt from the SHARDS rather than read from the artifact's own `arms` block --
        so a perturbed `arms` entry cannot launder itself into the bound that is checked
        against it.
        """

        document = _document()

        def span_from_shards(names: list[str]) -> dict[str, int]:
            reports = [
                json.loads((REPO / name).read_text(encoding="utf-8")) for name in names
            ]
            return {
                "games": sum(r["games"] for r in reports),
                "boundaries_measured": sum(r["boundaries_measured"] for r in reports),
                "transitions_diverged": sum(r["transitions_diverged"] for r in reports),
            }

        strict_shards = document["arms"]["strict"]["shards"]
        banded_shards = document["arms"]["banded"]["shards"]
        spans = {
            "strict": span_from_shards(strict_shards),
            "banded": span_from_shards(banded_shards),
            # DERIVED, and it must be every shard exactly once -- not the sum of two
            # artifact fields, which is what the arithmetic being checked is made of.
            "combined": span_from_shards(strict_shards + banded_shards),
        }
        self.assertEqual(
            sorted(document["statistical_bounds"]),
            sorted(spans),
            "the artifact's statistical_bounds no longer covers exactly strict, banded "
            "and combined; a missing arm is a bound nothing recomputes",
        )

        for arm in sorted(spans):
            span = spans[arm]
            with self.subTest(arm=arm):
                bounds = document["statistical_bounds"][arm]
                self.assertEqual(bounds["games"], span["games"])
                self.assertEqual(
                    bounds["boundaries_measured"], span["boundaries_measured"]
                )
                self.assertAlmostEqual(
                    bounds["rule_of_three_per_game_upper_95"], 3 / span["games"], places=8
                )
                self.assertAlmostEqual(
                    bounds["rule_of_three_per_boundary_upper_95"],
                    3 / span["boundaries_measured"],
                    places=10,
                )
                # ⚠ The denominator for a `divergence_class` negative is DIVERGENCES, not
                # boundaries: `classify_divergence` only runs on a boundary that already
                # diverged. Pinned separately, and now on `combined` too, because that is
                # the one the ledger quotes.
                self.assertEqual(
                    bounds["classified_divergences"], span["transitions_diverged"]
                )
                self.assertAlmostEqual(
                    bounds["rule_of_three_per_divergence_upper_95"],
                    3 / span["transitions_diverged"],
                    places=6,
                )
                self.assertEqual(bounds["one_in_n_games"], round(span["games"] / 3))
                self.assertEqual(
                    bounds["one_in_n_boundaries"],
                    round(span["boundaries_measured"] / 3),
                )

        # And the artifact's own `arms` block, against the same primary data -- the two
        # scalars the ledger's "25.8x" and "803,264" sentences are read off.
        for arm in ("strict", "banded"):
            with self.subTest(arm=arm, block="arms"):
                for field in ("games", "boundaries_measured", "transitions_diverged"):
                    self.assertEqual(document["arms"][arm][field], spans[arm][field])


class WhatTheCensusCannotSettleIsNamedTests(unittest.TestCase):
    """Requirement, not decoration: an audit already reversed one "closed" that was a
    fourth category in disguise. A zero produced by an instrument that could never have
    produced a one is not the same measurement as a zero produced by one that could."""

    def test_every_cited_line_number_still_points_at_what_it_claims(self) -> None:
        """Re-resolve the demonstrations' anchors against the tree as it is now.

        ⚠ ADDED AT THE MERGE OF `origin/main` AT `49c31855`. #1202 touched
        `engine_world.py`, `local_showdown.py` and `engine_search.py`, and **fifteen line
        citations in these demonstrations went stale in that one merge** -- the
        precounts raise 1958 -> 1971, `unsplit_legacy` 1986 -> 1999,
        `_public_effect_signals`'s two `blocked` branches 2391/2409 -> 2524/2542, the
        `deferredOpponentActions` emit 2350 -> 2372. Demonstrations whose entire value is
        that they were traced rather than asserted, invalidated by a merge, on the change
        whose subject is exactly that.

        The generator now resolves every number at generation time (`_anchor`,
        `_anchor_after`, `_raise_line`), so a moved anchor follows the code and a deleted
        one raises. This pin is the other half: it re-resolves them HERE and compares to
        what the committed artifact recorded, so an artifact generated against an older
        tree and not regenerated is red rather than quietly wrong.
        """

        recorded = _document()["verdicts"]
        for name, demonstration in sorted(CENSUS_SCRIPT.CENSUS_CANNOT_REACH.items()):
            with self.subTest(entry=name):
                self.assertEqual(recorded[name]["census_cannot_reach"], demonstration)
        for name, demonstration in sorted(
            CENSUS_SCRIPT.STRUCTURAL_DIVERGENCE_CLASSES.items()
        ):
            key = f"divergence_class:{name}"
            with self.subTest(entry=key):
                self.assertEqual(recorded[key]["structural_demonstration"], demonstration)
        # Anti-vacuity: the demonstrations must actually carry resolved numbers, or the
        # comparison above is between two identical pieces of prose and proves nothing.
        cited = re.findall(
            r":(\d{2,5})`",
            " ".join(CENSUS_SCRIPT.CENSUS_CANNOT_REACH.values())
            + " ".join(CENSUS_SCRIPT.STRUCTURAL_DIVERGENCE_CLASSES.values()),
        )
        self.assertGreater(len(cited), 20, "the demonstrations carry no line citations")

    def test_every_unreachable_entry_carries_its_demonstration(self) -> None:
        demonstrated = 0
        for name, record in sorted(_document()["verdicts"].items()):
            if record["verdict"] == "UNREACHABLE_STRUCTURAL":
                demonstrated += 1
                with self.subTest(entry=name):
                    self.assertIn("structural_demonstration", record)
                    self.assertGreater(len(record["structural_demonstration"]), 80)
        self.assertEqual(
            demonstrated,
            2,
            "§3.5 counts exactly two structurally unreachable divergence classes, "
            "`mapper_lossy` and `no_usable_branch`",
        )

    def test_the_entries_the_instrument_cannot_reach_are_recorded_as_such(self) -> None:
        document = _document()
        annotated = {
            name
            for name, record in document["verdicts"].items()
            if "census_cannot_reach" in record
        }
        self.assertEqual(
            annotated,
            set(CENSUS_SCRIPT.CENSUS_CANNOT_REACH),
            "the committed census and the generator disagree about which entries this "
            "instrument cannot reach",
        )
        self.assertTrue(
            annotated,
            "no entry is marked unreachable-by-instrument; the generator's own map is "
            "empty, which would make this vacuous",
        )
        for name in sorted(annotated):
            with self.subTest(entry=name):
                self.assertNotEqual(
                    document["verdicts"][name]["verdict"],
                    "FIRED",
                    f"{name} is annotated as unreachable by this instrument and fired "
                    "anyway; the annotation is wrong and must be removed, not kept",
                )


class TheEmissionGranularitySplitIsDerivedTests(unittest.TestCase):
    """Which of the three bounds applies to an entry is now AST-derived, and pinned.

    ⚠ ADDED AFTER REVIEW. The report split the 46 window-scoped entries into "40
    per-boundary refusal counters plus 6 per-game abort/error counters" and used that
    split to tell a reader which rule-of-three bound to apply. It was never traced, and
    **five of the six were wrong**: the three `engine_error*` keys increment inside the
    step `while` (per boundary), and `strict:no_damage_rolls` /
    `strict:branch_events_error:` increment inside `evaluate_boundary_strict`'s
    `for state in states` (per state within a boundary) -- which the differential's own
    comment at :3134-3136 says verbatim, in the block §6 and H8 both cite.

    Only `abort:no_legal_action` is per-game, and not because of loop depth: it sits two
    loops deep and is per-game because the next statement returns out of `run_game`. A
    depth heuristic would have got it wrong in the other direction, so the derivation
    detects the return structurally.

    Consequence, and the reason this is pinned rather than corrected once: the split
    decides a bound that differs by ~80x (3/8,000 against 3/641,866), and it was the only
    statement in the report of which entries take which. A plausible sentence about
    emission sites is exactly what `CENSUS_CANNOT_REACH`'s own rule forbids.

    MUTATION CHECKED: moving any entry between granularity buckets in the artifact, or
    perturbing a recorded bound away from 3/trials, turns this red.
    """

    def _window_scoped(self) -> dict[str, dict]:
        return {
            name: record
            for name, record in _document()["verdicts"].items()
            if record["family"].startswith("section_3_5")
            and not record.get("measurement_independent")
        }

    def test_the_split_is_exactly_one_per_game_counter(self) -> None:
        buckets = _document()["emission_granularity"]
        self.assertEqual(
            buckets.get("per_game"),
            ["abort:no_legal_action"],
            "the per-game bucket moved. Only `abort:no_legal_action` returns out of "
            "`run_game`; everything else in the inventory fires at least once per "
            "boundary, and quoting it the per-game bound understates the census by ~80x.",
        )
        self.assertEqual(sum(len(v) for v in buckets.values()), 46)
        self.assertEqual(
            sum(len(v) for k, v in buckets.items() if k != "per_game"),
            45,
        )

    def test_every_window_scoped_entry_resolves_to_an_emission_site(self) -> None:
        # An entry the AST cannot resolve has no derivable denominator, so its bound
        # would be a guess. That must be red, not silently defaulted.
        unresolved = sorted(
            name
            for name, record in self._window_scoped().items()
            if record["granularity"] == "UNRESOLVED" or not record.get("sites")
        )
        self.assertEqual(unresolved, [])

    def test_the_recorded_split_matches_a_live_ast_derivation(self) -> None:
        sites = CENSUS_SCRIPT.emission_sites()
        entries = CENSUS_SCRIPT.inventory()
        for name, record in sorted(self._window_scoped().items()):
            with self.subTest(entry=name):
                derived = CENSUS_SCRIPT.granularity(entries[name], sites)
                self.assertEqual(record["granularity"], derived["granularity"])
                self.assertEqual(record["denominator"], derived["denominator"])
                self.assertEqual(record["sites"], derived["sites"])

    def test_each_bound_is_three_over_its_own_denominator_from_the_shards(self) -> None:
        document = _document()
        shards = [
            json.loads((REPO / name).read_text(encoding="utf-8"))
            for name in document["arms"]["strict"]["shards"]
        ]
        trials = {
            "games": sum(r["games"] for r in shards),
            "boundaries_measured": sum(r["boundaries_measured"] for r in shards),
            "boundaries_full_round": sum(r["boundaries_full_round"] for r in shards),
        }
        # The three are genuinely different, so a bound cannot be right by coincidence.
        self.assertEqual(len(set(trials.values())), 3)
        recorded = document["denominator_trials"]
        self.assertEqual(sorted(recorded), sorted(trials))
        for name, value in sorted(trials.items()):
            with self.subTest(denominator=name):
                self.assertEqual(recorded[name]["trials"], value)
                self.assertAlmostEqual(
                    recorded[name]["rule_of_three_upper_95"], 3 / value, places=10
                )
        # Every entry names one of the three, so the join is total. The numbers live in
        # `denominator_trials` rather than on the entry ON PURPOSE: a number under a path
        # containing a counter name makes `tests/test_never_fired_counter_census.py` read
        # that counter as FIRED, and a first revision of this artifact did exactly that to
        # all 46.
        #
        # ⚠ WHAT THE TWO `assertNotIn`s BELOW ARE, STATED HONESTLY. A first draft of this
        # comment said they "keep the shape". They do not. Review replayed three variants:
        # `bound_trials` back on the record -> red here; a DIFFERENTLY NAMED numeric field
        # (`expected_hits_at_c152_rate`) -> green; a NESTED numeric sub-object
        # (`power.trials`) -> green. These two lines are a two-name regression pin against
        # the exact defect that shipped, not the invariant.
        #
        # THE INVARIANT IS ENFORCED BY A DIFFERENT MODULE, and that is deliberate design
        # rather than luck: `tests/test_never_fired_counter_census.py` matches on SHAPE --
        # any nonzero number under a dotted path containing a counter name -- so it caught
        # all three variants. RESIDUAL RISK, recorded because it is currently unstated
        # anywhere else: that guard holds only while this derived artifact stays inside
        # `counter_artifacts()`'s glob (`reports/` and `docs/`, recursive). A future census
        # written to `tests/data/` would leave the glob and lose the protection silently,
        # with nothing here noticing. See the cross-instrument note in this module's
        # docstring.
        self.assertTrue(
            str(CENSUS.relative_to(REPO)).startswith(("reports/", "docs/")),
            "the derived census left the corpus glob that enforces the "
            "no-numbers-under-a-counter-name invariant; see the residual-risk note above",
        )
        for name, record in sorted(self._window_scoped().items()):
            with self.subTest(entry=name):
                self.assertIn(record["denominator"], trials)
                self.assertNotIn("bound_trials", record)
                self.assertNotIn("rule_of_three_upper_95", record)

    def test_the_liveness_witness_map_is_derived_and_not_vacuous(self) -> None:
        """Emission-path liveness is what a calibrator establishes; pin how it was got.

        The report's earlier claim that six entries had "no in-family liveness witness"
        was false for three of them, so the witness relation is now computed rather than
        asserted -- and it is deliberately narrow: same emission statement, adjacent
        statement in one block, or strictly nested inside the entry's own loop. An
        over-broad relation would let a different except-handler vouch for an entry, and
        the first draft of it did exactly that (41 witnessed instead of 38).
        """

        document = _document()
        sites = CENSUS_SCRIPT.emission_sites()
        entries = CENSUS_SCRIPT.inventory()
        fired: dict[str, int] = {}
        for arm in document["arms"].values():
            for shard in arm["shards"]:
                for key, value in (
                    json.loads((REPO / shard).read_text(encoding="utf-8")).get("counters") or {}
                ).items():
                    if isinstance(value, (int, float)) and value:
                        fired[key] = fired.get(key, 0) + value
        derived = CENSUS_SCRIPT.liveness_witnesses(entries, sites, fired)
        window_scoped = self._window_scoped()
        for name, record in sorted(window_scoped.items()):
            with self.subTest(entry=name):
                self.assertEqual(record["liveness_witnesses"], derived.get(name, []))
        witnessed = set(document["window_scoped_with_a_liveness_witness"])
        self.assertEqual(
            witnessed,
            {n for n, r in window_scoped.items() if r["liveness_witnesses"]},
        )
        # Neither vacuous nor total: a relation that witnessed everything would be as
        # useless as one that witnessed nothing, and the unwitnessed set is what the
        # report has to be honest about.
        self.assertGreater(len(witnessed), 30)
        self.assertLess(len(witnessed), len(window_scoped))


class TheCannotReachSectionOfTheReportIsInSyncTests(unittest.TestCase):
    """§6 of the report is prose about `CENSUS_CANNOT_REACH`. Re-derive it from the map.

    ⚠ ADDED AFTER REVIEW, because §6 went out of sync in the very commit that fixed the
    map it describes. The generator and the artifact were corrected -- `public_effect_blocked`
    moved out, two more demonstrations re-traced -- and this narrative section was not. It
    kept asserting that entry unreachable *"because the differential declares none"*, a
    sentence the generator marks FALSE in capitals eleven files away in the same commit,
    and kept three further phrases the generator had just retracted. Meanwhile the ledger
    cross-reference added in that commit said SIX and §6 said SEVEN.

    Nothing caught it: the pin above asserts `artifact == generator`, and no pin read the
    prose. That is B1's shape -- an unpinned narrative count -- in the section whose entire
    job is to say what the census cannot settle.

    What is pinned is membership and arithmetic, not wording: the headline count, the two
    subsection counts, and the exact set of entries listed. Wording stays free so a
    demonstration can be improved without a test edit; a name silently joining or leaving
    cannot.

    MUTATION CHECKED: changing "Six" to "Seven", either subsection count, deleting a bullet,
    or re-listing `public_effect_blocked` among them turns this red.
    """

    _WORDS = {"Two": 2, "Three": 3, "Four": 4, "Five": 5, "Six": 6, "Seven": 7, "Eight": 8}

    def _report(self) -> str:
        return (REPO / "reports/c153_wide_seed_negative_census.md").read_text(encoding="utf-8")

    def _section_six(self) -> str:
        text = self._report()
        start = text.index("## 6. What this census cannot settle, named")
        return text[start : text.index("\n## ", start + 1)]

    def _listed(self) -> list[str]:
        body = self._section_six()
        listed_block = body[
            body.index("**Structural") : body.index("**Not in this category")
        ]
        return re.findall(r"^\* `([a-z0-9_]+)`", listed_block, re.M)

    def test_the_headline_count_matches_the_generator(self) -> None:
        stated = re.search(
            r"\*\*(" + "|".join(self._WORDS) + r")\*\* entries carry a "
            r"`census_cannot_reach` note",
            self._section_six(),
        )
        self.assertIsNotNone(
            stated,
            "§6's headline count sentence is gone or reworded; this pin reads it and must "
            "be updated with it rather than silently passing",
        )
        self.assertEqual(
            self._WORDS[stated.group(1)],
            len(CENSUS_SCRIPT.CENSUS_CANNOT_REACH),
            f"§6 says {stated.group(1)} entries carry the note; the generator has "
            f"{len(CENSUS_SCRIPT.CENSUS_CANNOT_REACH)}",
        )

    def test_the_two_subsection_counts_match_the_generator(self) -> None:
        body = self._section_six()
        structural = {
            name
            for name, entry in CENSUS_SCRIPT.inventory().items()
            if "structural_demonstration" in entry
            and name in CENSUS_SCRIPT.CENSUS_CANNOT_REACH
        }
        expected_structural = len(structural)
        expected_input = len(CENSUS_SCRIPT.CENSUS_CANNOT_REACH) - expected_structural
        for label, expected in (
            (r"\*\*Structural — not measurement results at all \((\d+)\)\.\*\*", expected_structural),
            (
                r"\*\*Reachable only by an input this instrument does not build \((\d+)\)\.\*\*",
                expected_input,
            ),
        ):
            found = re.search(label, body)
            self.assertIsNotNone(found, f"§6 subsection heading {label!r} is gone")
            with self.subTest(heading=label):
                self.assertEqual(int(found.group(1)), expected)

    def test_the_listed_entries_are_exactly_the_generators(self) -> None:
        expected = {name.rsplit(":", 1)[-1] for name in CENSUS_SCRIPT.CENSUS_CANNOT_REACH}
        self.assertEqual(
            set(self._listed()),
            expected,
            "§6 lists a different set of unreachable entries than the generator carries",
        )
        self.assertEqual(
            len(self._listed()),
            len(expected),
            "§6 lists an entry twice",
        )

    # Spans on a line that mark text as QUOTED rather than asserted: a GFM emphasis-quote
    # `*"..."*`, a plain double-quoted run, or a code span. Deliberately not a proximity
    # window -- see the docstring below for the smuggling that killed the window version.
    _QUOTED_SPAN = re.compile(r'\*"[^"]*"\*|"[^"]*"|`[^`]*`')

    @staticmethod
    def _blocks(path: Path) -> list[tuple[int, str]]:
        """`(first line number, whitespace-normalised block)` per markdown block.

        ⚠ NOT per physical line, and the difference is not cosmetic. Both documents are
        hard-wrapped at ~95 columns, so a quoted retraction routinely straddles a line
        break: the rest-sleep one reads `"live` at the end of one line and
        `rows always carry the counts"` at the start of the next. A per-line matcher never
        sees that phrase AT ALL -- it passes vacuously, which is the same blind spot in the
        other direction, and an unquoted assertion wrapped the same way would also escape.

        Blocks are separated by blank lines and by list-item starts, so a bullet is its own
        block and a quoted span cannot be borrowed from the paragraph above it.
        """

        blocks: list[tuple[int, str]] = []
        current: list[str] = []
        start = 1
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            begins_item = line.lstrip().startswith(("* ", "- ", "| ")) or not line.strip()
            if begins_item and current:
                blocks.append((start, " ".join(current)))
                current = []
            if not line.strip():
                continue
            if not current:
                start = number
            current.append(line.strip())
        if current:
            blocks.append((start, " ".join(current)))
        return blocks

    def test_no_adjudicated_false_phrasing_is_asserted_anywhere_in_the_report(self) -> None:
        """A negative pin over the four sentences review has adjudicated FALSE.

        ⚠ ADDED AFTER REVIEW POINTED OUT WHAT THE COUNT-AND-MEMBERSHIP PIN MISSES. Of the
        four §6 defects that shipped at `f4ad9802`, only ONE was a membership change
        (`public_effect_blocked` moving out). The other three were retracted PHRASINGS
        with the entry unmoved -- so the pin above, which checks counts and membership,
        would have caught 1 of 4.

        This is not a wording checksum, and deliberately not: a checksum would fight
        exactly the demonstration improvements this section keeps getting. It is four
        specific substrings that review has ruled false, in the repo's existing
        `test_prose_alone_is_not_evidence` shape -- it constrains nothing future and
        forbids only the return of a claim already adjudicated.

        ⚠ THE EXEMPTION IS QUOTING, NOT PROXIMITY, AND THE FIRST VERSION HAD IT WRONG.
        That version exempted any line within a few lines of a `⚠` / `FALSE` / `retract`
        marker, on the reasoning that these documents hard-wrap and a retraction and its
        quoted phrase land on different lines. **Every §6 bullet opens with `⚠`**, so the
        window covered precisely the paragraphs the pin exists to protect: review smuggled
        "the public-materialization payload never carries them at all" into the
        `deferred_opponent_action` bullet and "live rows always carry the counts" into the
        rest-sleep bullet, and both went green. A bare housekeeping `⚠` two lines above any
        assertion exempted it anywhere -- 15 % of this report's lines and 25 % of the
        ledger's. The docstring's own "what is forbidden is a line asserting it" was false
        as implemented.

        So the rule is now a property of the OCCURRENCE, not of its neighbourhood: the
        phrase must sit inside quote marks on the line that carries it -- `*"..."*`, a
        plain double-quoted run, or a code span. Every legitimate retraction in both
        documents already writes it that way, so this is strictly stricter and passes the
        text unchanged.
        """

        retracted = (
            "the differential declares none",
            "never carries",
            "nowhere in the repository",
            "live rows always carry the counts",
        )
        offenders = []
        for path in (
            "reports/c153_wide_seed_negative_census.md",
            "reports/c138_known_gaps_ledger.md",
        ):
            for number, line in self._blocks(REPO / path):
                quoted = [m.span() for m in self._QUOTED_SPAN.finditer(line)]
                # CASE-INSENSITIVE. One of review's three smuggles passed the first
                # version of this pin on a capital letter alone -- "Live rows always
                # carry the counts" at the head of a sentence. A retracted claim is not
                # a different claim for being sentence-cased.
                haystack = line.casefold()
                for phrase in retracted:
                    needle = phrase.casefold()
                    start = haystack.find(needle)
                    while start != -1:
                        end = start + len(needle)
                        inside = any(lo <= start and end <= hi for lo, hi in quoted)
                        if not inside:
                            offenders.append(
                                f"{path}:{number}: {phrase!r} asserted unquoted -- "
                                f"{line.strip()[:90]}"
                            )
                            break
                        start = haystack.find(needle, end)
        self.assertEqual(
            offenders,
            [],
            "a phrasing review has adjudicated FALSE is asserted outside quote marks. "
            "Quote it as retracted, or do not write it:\n" + "\n".join(offenders),
        )

    def test_the_quoting_exemption_cannot_be_smuggled(self) -> None:
        """The three smuggles that passed the window version, as fixtures.

        A negative pin is only worth its exemption rule, and this one's first exemption
        rule was defeated by the documents' own house style. These are the exact strings
        review used; each must be seen as an unquoted assertion.
        """

        smuggles = (
            "⚠ Housekeeping note.",  # the bare marker that used to exempt a whole window
            "the public-materialization payload never carries them at all, so the guard "
            "cannot fire.",
            "Every live row always carry the counts, so the branch is dead.",
        )
        quoted_forms = (
            'the earlier *"never carries"* is retracted',
            'a draft said "live rows always carry the counts" and it was wrong',
            'the sentence `the differential declares none` is false',
        )
        for text in smuggles[1:]:
            with self.subTest(smuggle=text[:40]):
                spans = [m.span() for m in self._QUOTED_SPAN.finditer(text)]
                phrase = "never carries" if "never carries" in text else "always carry the counts"
                start = text.find(phrase)
                self.assertFalse(
                    any(lo <= start and start + len(phrase) <= hi for lo, hi in spans),
                    f"{text!r} would still be exempt, so the pin is smugglable",
                )
        for text in quoted_forms:
            with self.subTest(legitimate=text[:40]):
                spans = [m.span() for m in self._QUOTED_SPAN.finditer(text)]
                self.assertTrue(spans, f"{text!r} has no quoted span; a real retraction must")

    def test_public_effect_blocked_is_named_as_reachable_and_not_listed(self) -> None:
        # The specific correction, pinned by name so it cannot come back as prose. It must
        # appear in §6 -- silently deleting it would hide the retraction -- but never among
        # the bullets, and never in the generator's map.
        body = self._section_six()
        self.assertIn("public_effect_blocked", body)
        self.assertNotIn("public_effect_blocked", self._listed())
        self.assertNotIn(
            "skip:world_unsupported:public_effect_blocked",
            CENSUS_SCRIPT.CENSUS_CANNOT_REACH,
        )
        self.assertEqual(
            _document()["verdicts"]["skip:world_unsupported:public_effect_blocked"]["verdict"],
            "NOT_OBSERVED_AT_SCOPE",
        )


class C152sRefutationsNowHaveAReproducibleBuildTests(unittest.TestCase):
    """C152 refuted two "never fired" claims on an UNREPRODUCIBLE engine.

    `reports/artifacts/c152_wide_census_*_sweep.json` were taken at fingerprint
    `89797289...` -- a throwaway instrumented build, the shipping tree plus two
    `eprintln!` blocks, which `tests/test_harness_digest_provenance.py` records as "NOT
    reproducible from any committed tree, by design". Until this census, the entire
    evidence for `skip:rump_branch_set` and
    `strict:branch_event_legal_error:BranchLegalRollError` firing at all lived on an engine
    nobody can rebuild. These pins move it onto `bfdbe1c04876edcd`, which anyone can.
    """

    def test_both_counters_fire_on_the_shipping_build(self) -> None:
        recorded = _document()["c152_refutations_on_the_shipping_build"]
        self.assertEqual(
            sorted(recorded),
            [
                "skip:rump_branch_set",
                "strict:branch_event_legal_error:BranchLegalRollError",
            ],
        )
        for key, arms in sorted(recorded.items()):
            with self.subTest(counter=key):
                self.assertGreater(
                    arms["strict_arm"],
                    0,
                    f"{key} did not fire on the STRICT arm of the shipping build. C152's "
                    "refutation would then rest entirely on an engine that cannot be "
                    "rebuilt, and §3.5's correction would be unsupported.",
                )

    def test_the_figures_re_derive_from_the_committed_shards(self) -> None:
        document = _document()
        for arm_name, field in (("strict", "strict_arm"), ("banded", "banded_arm")):
            shards = [
                json.loads((REPO / name).read_text(encoding="utf-8"))
                for name in document["arms"][arm_name]["shards"]
            ]
            for key, arms in sorted(
                document["c152_refutations_on_the_shipping_build"].items()
            ):
                with self.subTest(counter=key, arm=arm_name):
                    self.assertEqual(
                        sum((r.get("counters") or {}).get(key, 0) for r in shards),
                        arms[field],
                    )


class TheAbsencePinsAreNotVacuousTests(unittest.TestCase):
    def test_the_controls_fire(self) -> None:
        # Four counters that DO fire in this census. Every absence assertion above is a
        # loop over the same twelve shards, and a loop over a shard set that stopped being
        # read passes; these are what make the loops mean something.
        controls = _document()["controls"]
        self.assertEqual(
            sorted(controls),
            [
                "skip:unmappable_choice:struggle_not_submittable",
                "skip:world_unsupported:materialization_blocker",
                "skip:world_unsupported:volatile_unsupported",
                "world_prestate_mismatch",
            ],
        )
        for key, value in sorted(controls.items()):
            with self.subTest(control=key):
                self.assertGreater(value, 0, f"the control {key} did not fire")

    def test_the_controls_are_re_derived_from_the_shards_not_transcribed(self) -> None:
        document = _document()
        shards = [
            json.loads((REPO / name).read_text(encoding="utf-8"))
            for arm in document["arms"].values()
            for name in arm["shards"]
        ]
        for key, value in sorted(document["controls"].items()):
            with self.subTest(control=key):
                self.assertEqual(
                    sum((report.get("counters") or {}).get(key, 0) for report in shards),
                    value,
                )

    def test_the_shard_list_is_non_empty_and_every_member_exists(self) -> None:
        document = _document()
        names = [name for arm in document["arms"].values() for name in arm["shards"]]
        self.assertEqual(len(names), 12)
        for name in names:
            with self.subTest(shard=name):
                self.assertTrue((REPO / name).is_file(), f"{name} is not committed")

    def test_the_verdict_partition_of_every_shard_closes(self) -> None:
        # The five-term identity C144 established, re-derived on this census's own
        # shards. A shard whose partition does not close is not a measurement of
        # anything, and every absence above would inherit the defect silently.
        document = _document()
        for name in [n for arm in document["arms"].values() for n in arm["shards"]]:
            report = json.loads((REPO / name).read_text(encoding="utf-8"))
            counters = report["counters"]
            with self.subTest(shard=name):
                self.assertEqual(
                    report["transitions_matched"]
                    + report["transitions_diverged"]
                    + report["engine_errors"]
                    + counters.get("skip:strict_all_branches_lossy", 0)
                    + counters.get("skip:rump_branch_set", 0),
                    report["boundaries_measured"],
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
