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
added to it must be added here too.

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

⚠ 13 was found by the reviewer, not the author, and it was the important one: `combined`
is where the 0.32 % bound quoted in the report, in §3.5 and in H15's cell comes from, and
the first revision of `test_the_stated_bounds_...` looped over `document["arms"]`, which
holds only `strict` and `banded`. Four separate mutations of the combined block passed
green. Three of the twelve above (3, 8 and 13's neighbours) were likewise near-misses at
first: 3 survived until the closure pin stopped reading the artifact's own `agrees` flag,
and 8's first form was a defective mutation (`{} or {...}`, which is truthy) rather than a
surviving pin. Both are recorded rather than tidied away.

WHAT IS NOT PINNED, ON PURPOSE. Nothing here asserts a divergence rate. These are
unregistered seeds and the census is not fidelity evidence; §7.3 of
`reports/c152_ledger_terminal_disposition.md` says so about its own predecessor and the
same applies with more force to a sample twenty-five times the size of the two windows.
"""

from __future__ import annotations

import importlib.util
import json
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
