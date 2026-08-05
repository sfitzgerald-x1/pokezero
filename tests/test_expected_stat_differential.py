"""V2 — ground-truth differential for the ``EXPECTED_{DEF,SPA,SPD,SPE}`` columns.

The audit plan's words about this block: "asserted exact and have never been differentially
checked." Both stat defects found in this generation (C1's spread fork, the L100 zeroing) lived
in the neighbouring HP/Atk columns, so the four remaining ones are the same family with the worst
record and no coverage.

What this file establishes, and the order matters:

1. the encoder's four columns agree with the GENERATOR CORE (``randbats_spread_details``) for
   every pool set at its real level, and across levels 1..100 as the off-pool sweep;
2. the core is not merely a second Python opinion -- ``scripts/expected_stat_gate.py`` anchors it
   to ENGINE-computed stats read from the opposing seat's own opening ``|request|``, which is the
   channel ``scripts/investment_gate.py`` already cross-checks it on. A Python-vs-Python
   differential whose two sides share a bug proves nothing (plan §3), so the chain is
   encoder -> core -> engine, with this file owning the first link and the gate the second;
3. the specific defect this item found stays dead: Def/SpA/SpD/Spe were computed at a flat
   ``iv=31``, ignoring the generator's Hidden Power ``HPivs`` override. Scope, at the variant
   level (the shape that actually exists): **716 of 1682 variants, 42.6%**, carry a def/spa/spd/spe
   override -- def 312, spa 417, spd 334, spe 145. (An earlier revision of this line said "205 of
   393 pool sets"; 205 is the number of sets.json ROWS carrying an override, not a count of wrong
   values, and a sets.json row is a whole movepool -- a shape no real variant has. The cell-level
   figure on that surface is 348 of 820.) ``test_flat_iv31_encoder_is_killed`` is the kill-confirmed
   mutation for that -- a guard nobody has broken on purpose is not measured coverage (plan §3).

Reachability is asserted, not assumed: a sweep that never reached a Hidden-Power set would pass
vacuously, which is the bug and not the fix.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Iterator, Mapping

from pokezero.belief import RevealedPokemonBelief
from pokezero.dex import load_showdown_dex_cached
from pokezero.gen3_damage import (
    HIDDEN_POWER_IVS,
    hidden_power_type,
    randbats_spread_details,
)
from pokezero.showdown import (
    NUMERIC_EXPECTED_DEF,
    NUMERIC_EXPECTED_SPA,
    NUMERIC_EXPECTED_SPD,
    NUMERIC_EXPECTED_SPE,
    _V4_NUMERIC_FEATURE_COUNT,
    _ACTUAL_STAT_DIVISOR,
    _encode_expected_stats,
    _gen3_stat,
)

from ._showdown_root import requires_showdown, showdown_root

# The four columns under test, paired with the generator's stat key.
COLUMNS: tuple[tuple[str, int], ...] = (
    ("def", NUMERIC_EXPECTED_DEF),
    ("spa", NUMERIC_EXPECTED_SPA),
    ("spd", NUMERIC_EXPECTED_SPD),
    ("spe", NUMERIC_EXPECTED_SPE),
)

# Levels swept off-pool. The pool's own levels are covered separately at each species' real
# level; this arm exists because the L100 zeroing defect was a LEVEL-shape bug that no fixture
# carried, so the level axis gets swept independently of what the pool happens to use.
OFF_POOL_LEVELS = tuple(range(1, 101))


def _pool_sets(root: Path) -> Iterator[tuple[str, int, Mapping[str, Any]]]:
    """(species_key, real_level, set_row) over every gen3 randbats pool set."""
    sets = json.loads((root / "data" / "random-battles" / "gen3" / "sets.json").read_text())
    for species, entry in sets.items():
        level = int(entry.get("level") or 100)
        for row in entry.get("sets", []):
            yield species, level, row


def _variant_from_set(row: Mapping[str, Any]) -> dict[str, Any]:
    """The candidate-variant payload shape the encoder consumes, from a sets.json row."""
    items = row.get("items") or [None]
    return {"moves": list(row.get("movepool", [])), "item": items[0]}


def _encode(
    dex,
    *,
    species: str,
    level: int,
    variants: tuple[Mapping[str, Any], ...],
    base_species: str | None = None,
):
    """Run the real encoder block and return its four emitted column values.

    ``base_species`` defaults to ``species`` but is separable, because the forme arm needs
    ``base_species != battle_species`` to construct the "fell back to the base species" scenario
    at all. With them always equal that test could not fail no matter which map the encoder read.
    """
    num_row = [0.0] * _V4_NUMERIC_FEATURE_COUNT
    belief = RevealedPokemonBelief(
        showdown_slot="p2a",
        species=species,
        candidate_variants=variants,
    )
    _encode_expected_stats(
        num_row,
        dex,
        base_species=base_species or species,
        battle_species=species,
        details=f"{species}, L{level}",
        belief=belief,
        exact_spreads=True,
    )
    return num_row


def _expected_column_value(stat: int) -> float:
    return min(1.0, stat / _ACTUAL_STAT_DIVISOR)


@requires_showdown()
class ExpectedStatDifferentialTest(unittest.TestCase):
    """Encoder vs the generator core, over the whole pool and the whole level axis."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = showdown_root()
        cls.dex = load_showdown_dex_cached(cls.root)
        cls.pool = tuple(_pool_sets(cls.root))

    def _truth(self, species: str, level: int, row: Mapping[str, Any]) -> Mapping[str, int]:
        info = self.dex.species_info(species)
        assert info is not None
        items = row.get("items") or [None]
        # has_physical_attack is irrelevant to these four columns (Atk zeroing touches atk, the
        # trim loops touch hp), so either arm gives the same Def/SpA/SpD/Spe. Passing True keeps
        # the call honest rather than inventing a category count this assertion does not use.
        return randbats_spread_details(
            info.base_stats,
            level=level,
            moves=list(row.get("movepool", [])),
            item=items[0],
            has_physical_attack=True,
        ).stats

    def test_pool_sets_at_real_level_match_the_generator_core(self) -> None:
        """Every pool set, pinned to one candidate, at its real level: exact agreement."""
        checked = 0
        override_sets = 0
        for species, level, row in self.pool:
            info = self.dex.species_info(species)
            if info is None:
                continue
            truth = self._truth(species, level, row)
            num_row = _encode(
                self.dex,
                species=species,
                level=level,
                variants=(_variant_from_set(row),),
            )
            hp_type = hidden_power_type(list(row.get("movepool", [])))
            override = HIDDEN_POWER_IVS.get(hp_type or "", {})
            if any(stat in override for stat, _ in COLUMNS):
                override_sets += 1
            for stat, slot in COLUMNS:
                if not info.base_stats.get(stat):
                    continue
                with self.subTest(species=species, level=level, stat=stat):
                    self.assertAlmostEqual(
                        num_row[slot],
                        _expected_column_value(truth[stat]),
                        places=9,
                        msg=(
                            f"{species} L{level} {stat}: encoder emitted {num_row[slot]}, "
                            f"generator core says {truth[stat]} "
                            f"(hidden power type {hp_type!r}, IV override {dict(override)})"
                        ),
                    )
                checked += 1
        # Reachability preconditions (plan §3): this assertion is worthless if the sweep never
        # reached a set whose IVs the generator actually overrides. 205 was the measured count
        # at the time V2 was built; assert a floor rather than the exact number so a pool
        # update does not fail the guard for the wrong reason.
        self.assertGreater(checked, 1000, "pool sweep did not reach the pool")
        self.assertGreater(
            override_sets,
            150,
            "sweep reached almost no Hidden-Power-override sets; it would pass vacuously",
        )

    def test_every_real_candidate_variant_matches_the_generator_core(self) -> None:
        """The production surface: the 1682 variants the encoder is actually handed.

        The pool sweep above feeds a sets.json row's WHOLE MOVEPOOL as one variant's moves, which
        is not a shape the generator can produce (it picks four). That arm is still worth having --
        it sweeps every species and level cheaply -- but it validates spreads no real candidate
        has, so the real candidate universe gets its own exhaustive arm here.
        """
        from pokezero.randbat import load_gen3_randbat_source_cached

        source = load_gen3_randbat_source_cached(self.root)
        checked = 0
        override_variants = 0
        for universe in source.universes.values():
            info = self.dex.species_info(universe.species)
            if info is None:
                continue
            for entry in universe.variants:
                variant = {"moves": list(entry.moves), "item": entry.item}
                truth = randbats_spread_details(
                    info.base_stats,
                    level=entry.level,
                    moves=list(entry.moves),
                    item=entry.item,
                    has_physical_attack=True,
                ).stats
                num_row = _encode(
                    self.dex,
                    species=universe.species,
                    level=entry.level,
                    variants=(variant,),
                )
                hp_type = hidden_power_type(list(entry.moves))
                if any(stat in HIDDEN_POWER_IVS.get(hp_type or "", {}) for stat, _ in COLUMNS):
                    override_variants += 1
                for stat, slot in COLUMNS:
                    if not info.base_stats.get(stat):
                        continue
                    with self.subTest(variant=entry.variant_id, stat=stat):
                        self.assertAlmostEqual(
                            num_row[slot], _expected_column_value(truth[stat]), places=9
                        )
                    checked += 1
        # Reachability, measured: 1682 variants, 716 of them carrying an override.
        self.assertGreater(checked, 5000, "did not reach the real candidate universe")
        self.assertGreater(
            override_variants, 600, "too few overriding variants reached; arm would be weak"
        )

    def test_off_pool_level_sweep_matches_the_generator_core(self) -> None:
        """Levels 1..100 for a representative override set per HP type, plus L100 explicitly.

        The L100 zeroing defect was a level-shape bug live on three schema generations, found
        only because a parity harness was pointed at the one details shape no fixture carried.
        This sweeps the level axis directly instead of trusting the pool's own levels.
        """
        by_type: dict[str, tuple[str, Mapping[str, Any]]] = {}
        for species, _level, row in self.pool:
            hp_type = hidden_power_type(list(row.get("movepool", [])))
            if hp_type and hp_type not in by_type and HIDDEN_POWER_IVS.get(hp_type):
                by_type[hp_type] = (species, row)
        self.assertGreaterEqual(
            len(by_type), 10, "expected representatives for most Hidden Power types"
        )
        for hp_type, (species, row) in sorted(by_type.items()):
            info = self.dex.species_info(species)
            assert info is not None
            for level in OFF_POOL_LEVELS:
                truth = self._truth(species, level, row)
                num_row = _encode(
                    self.dex,
                    species=species,
                    level=level,
                    variants=(_variant_from_set(row),),
                )
                for stat, slot in COLUMNS:
                    if not info.base_stats.get(stat):
                        continue
                    with self.subTest(hp_type=hp_type, species=species, level=level, stat=stat):
                        self.assertAlmostEqual(
                            num_row[slot],
                            _expected_column_value(truth[stat]),
                            places=9,
                        )

    def test_level_100_details_shape_without_a_level_token(self) -> None:
        """``details`` carrying no ``L`` token means L100 -- the shape that hid the zeroing bug.

        Nine pool species encode with a token-less details string. A regression here does not
        produce a wrong number, it produces ELEVEN ZEROED CELLS, which is why it survived: an
        absent value looks like a masked feature, not like a defect.
        """
        overrides = [
            (species, level, row)
            for species, level, row in self.pool
            if HIDDEN_POWER_IVS.get(hidden_power_type(list(row.get("movepool", []))) or "", {})
        ]
        self.assertTrue(overrides, "no override sets found; precondition failed")
        for species, _level, row in overrides[:40]:
            info = self.dex.species_info(species)
            assert info is not None
            truth = self._truth(species, 100, row)
            num_row = [0.0] * _V4_NUMERIC_FEATURE_COUNT
            belief = RevealedPokemonBelief(
                showdown_slot="p2a",
                species=species,
                candidate_variants=(_variant_from_set(row),),
            )
            _encode_expected_stats(
                num_row,
                self.dex,
                base_species=species,
                battle_species=species,
                details=species,  # no ", L100" token at all
                belief=belief,
                exact_spreads=True,
            )
            for stat, slot in COLUMNS:
                if not info.base_stats.get(stat):
                    continue
                with self.subTest(species=species, stat=stat):
                    self.assertNotEqual(
                        num_row[slot], 0.0, "token-less details zeroed the stat block"
                    )
                    self.assertAlmostEqual(
                        num_row[slot], _expected_column_value(truth[stat]), places=9
                    )

    def test_formes_use_the_battle_species_base_stats(self) -> None:
        """Functional formes are distinct sets.json keys; each must use its OWN base stats.

        Deoxys' formes differ enormously in Def/SpA/SpD/Spe, so a forme that fell back to the
        base species would be wrong by a wide margin rather than by the one point the IV bug
        cost -- worth pinning separately from the pool sweep.
        """
        # sets.json keys are NORMALIZED ("deoxysattack", not "Deoxys-Attack"), so the forme
        # marker lives on the dex's display name, not the key. Asserting on the key found zero
        # formes and would have made this whole arm vacuous.
        def _is_forme(species: str) -> bool:
            info = self.dex.species_info(species)
            return info is not None and "-" in info.name

        formes = {species for species, _l, _r in self.pool if _is_forme(species)}
        self.assertTrue(formes, "no forme keys in the pool; precondition failed")
        seen = 0
        for species, level, row in self.pool:
            if not _is_forme(species):
                continue
            info = self.dex.species_info(species)
            if info is None:
                continue
            truth = self._truth(species, level, row)
            num_row = _encode(
                self.dex,
                species=species,
                level=level,
                variants=(_variant_from_set(row),),
            )
            for stat, slot in COLUMNS:
                if not info.base_stats.get(stat):
                    continue
                with self.subTest(forme=species, stat=stat):
                    self.assertAlmostEqual(
                        num_row[slot], _expected_column_value(truth[stat]), places=9
                    )
            seen += 1
        self.assertGreater(seen, 5, "forme arm reached too few formes to mean anything")

    def test_a_forme_does_not_read_the_base_species_stat_block(self) -> None:
        """The forme claim, actually constructed: ``base_species != battle_species``.

        The sweep above always passes the same species as both, so it could not distinguish
        "reads battle_species" from "reads base_species" -- mutating the encoder to read the base
        species' stats left it green. Deoxys' formes differ enormously on these four, so encoding
        Deoxys-Attack while naming plain Deoxys as the base species must still emit the ATTACK
        forme's numbers (base_species feeds only the HP baseline).
        """
        attack = self.dex.species_info("deoxysattack")
        base = self.dex.species_info("deoxys")
        assert attack is not None and base is not None
        differing = [
            stat
            for stat, _slot in COLUMNS
            if attack.base_stats.get(stat) != base.base_stats.get(stat)
        ]
        self.assertTrue(
            differing, "Deoxys-Attack and Deoxys agree on all four; pick another forme"
        )
        _species, _level, row = self.pool[0]
        variants = (_variant_from_set(row),)
        num_row = _encode(
            self.dex,
            species="deoxysattack",
            level=100,
            variants=variants,
            base_species="deoxys",
        )
        for stat, slot in COLUMNS:
            if stat not in differing:
                continue
            with self.subTest(stat=stat):
                want = _gen3_stat(int(attack.base_stats[stat]), 100, ev=85, iv=31, hp=False)
                wrong = _gen3_stat(int(base.base_stats[stat]), 100, ev=85, iv=31, hp=False)
                self.assertNotAlmostEqual(
                    num_row[slot], _expected_column_value(wrong), places=9
                )
                # The pool row's variant may carry a Hidden Power override, so assert the value is
                # derived from the ATTACK forme rather than pinning one exact number.
                self.assertLessEqual(num_row[slot], _expected_column_value(want) + 1e-12)

    def test_ambiguous_candidates_emit_the_upper_bound_never_a_midpoint(self) -> None:
        """With candidates that disagree, the column is the MAX -- a real bound, not a fiction.

        29 of 220 pool species have candidate sets that disagree on at least one of the four
        on the sets.json-row shape this test uses (60 disagreeing cells); on the 1682-variant
        surface, which is the shape that really occurs, it is 77 of 220. An earlier revision said
        30, which is neither.
        (different Hidden Power types, hence different IV overrides). A single-valued column
        cannot carry a band, so the contract is: exact when the candidates agree, and otherwise
        the maximum over them, which is a value some real candidate actually has. A midpoint or
        an average would be a number no candidate has -- the "degrade to absent, never to
        plausible" failure the audit exists to find.
        """
        by_species: dict[str, list[Mapping[str, Any]]] = {}
        levels: dict[str, int] = {}
        for species, level, row in self.pool:
            by_species.setdefault(species, []).append(row)
            levels[species] = level
        disagreeing = 0
        for species, rows in by_species.items():
            info = self.dex.species_info(species)
            if info is None or len(rows) < 2:
                continue
            level = levels[species]
            truths = [self._truth(species, level, row) for row in rows]
            variants = tuple(_variant_from_set(row) for row in rows)
            num_row = _encode(self.dex, species=species, level=level, variants=variants)
            for stat, slot in COLUMNS:
                if not info.base_stats.get(stat):
                    continue
                values = {truth[stat] for truth in truths}
                if len(values) > 1:
                    disagreeing += 1
                with self.subTest(species=species, stat=stat):
                    self.assertAlmostEqual(
                        num_row[slot],
                        _expected_column_value(max(values)),
                        places=9,
                        msg=f"{species} {stat}: candidates {sorted(values)}",
                    )
                    # The emitted value is one a real candidate has, always.
                    self.assertIn(
                        round(num_row[slot] * _ACTUAL_STAT_DIVISOR),
                        values,
                        "emitted a value no candidate variant has",
                    )
        self.assertGreater(
            disagreeing, 0, "no disagreeing candidate sets reached; assertion was vacuous"
        )

    def test_one_unevaluable_candidate_abandons_the_whole_set(self) -> None:
        """A max over a STRICT SUBSET of the candidates can fall BELOW the true value.

        That is unsound in exactly the direction this column claims to be safe in: the emitted
        value is only an upper bound if it was taken over ALL candidates. So one unevaluable
        candidate must abandon the set entirely and fall back to the flat iv=31 no-set-source
        state -- not skip that candidate and bound over the rest.

        Kill-confirmed guard: changing the encoder's ``break`` to ``continue`` passes every other
        test in this file, which is why this one exists. The fixture is built so the subset max
        and the flat value DIFFER, or the assertion could not tell them apart.
        """
        species = "meganium"
        info = self.dex.species_info(species)
        assert info is not None
        level = 100
        # Hidden Power Fighting overrides all four IVs to 30, so its spread is strictly below the
        # flat iv=31 value on every column under test.
        good = {"moves": ["hiddenpowerfighting", "gigadrain", "synthesis", "toxic"], "item": "leftovers"}
        broken = {"moves": None, "item": "leftovers"}

        flat = _encode(self.dex, species=species, level=level, variants=())
        subset = _encode(self.dex, species=species, level=level, variants=(good,))
        # Reachability: the two states must be distinguishable or this proves nothing.
        differs = [
            stat
            for stat, slot in COLUMNS
            if info.base_stats.get(stat) and flat[slot] != subset[slot]
        ]
        self.assertTrue(
            differs,
            "fixture does not separate the subset bound from the flat value; assertion vacuous",
        )

        # EVERY ORDER, and this is the point. The first version of this test only tried
        # (good, broken) -- with the unevaluable candidate LAST, which is the one order where a
        # `break` -> `continue` mutation is harmless: the list is cleared and never refilled, so
        # the flat fallback happens anyway and the mutation survives with the whole suite green.
        # Reversed, that same mutation emits a max over the strict subset {good}: measured
        # def/spa/spd/spe = 256/222/256/216 against the flat 257/223/257/217, i.e. BELOW the true
        # bound -- exactly the unsoundness this guard exists to prevent. Candidate order comes from
        # the belief engine, not from this test, so the guard has to hold in all of them.
        # The third variant must ALSO carry an IV override. An earlier draft used a plain
        # no-Hidden-Power set here, which made the arm VACUOUS: under the mutation the surviving
        # suffix maxes to exactly the flat value, so the assertion could not tell the two apart.
        # That is why this test reported 4 subtest failures under mutation rather than 8.
        good2 = {"moves": ["hiddenpowerice", "gigadrain", "synthesis", "toxic"], "item": "leftovers"}
        for label, variants in (
            ("broken last", (good, broken)),
            ("broken first", (broken, good)),
            ("broken middle", (good, broken, good2)),
        ):
            abandoned = _encode(self.dex, species=species, level=level, variants=variants)
            for stat, slot in COLUMNS:
                if not info.base_stats.get(stat):
                    continue
                with self.subTest(order=label, stat=stat):
                    self.assertAlmostEqual(
                        abandoned[slot],
                        flat[slot],
                        places=9,
                        msg=(
                            f"{stat} ({label}): one unevaluable candidate did not abandon the "
                            "set -- the emitted value is a max over a strict subset, which is "
                            "not a bound"
                        ),
                    )

    def test_flat_iv31_encoder_is_killed(self) -> None:
        """Kill-confirmed mutation: the pre-fix encoder must FAIL this file's differential.

        The mutation is the exact code that shipped -- ``_gen3_stat(base, level, ev=85, iv=31)``
        with no Hidden Power override. If this test cannot tell the two apart, the differential
        above is not measuring anything.

        This calls the REAL ENCODER for both arms. An earlier version recomputed the mutation
        inline and compared it against the core, which proved the two formulas differ but never
        ran the encoder at all -- it would have stayed green if the encoder stopped consuming
        candidate variants entirely.
        """
        killed = 0
        checked = 0
        for species, level, row in self.pool:
            info = self.dex.species_info(species)
            if info is None:
                continue
            hp_type = hidden_power_type(list(row.get("movepool", [])))
            override = HIDDEN_POWER_IVS.get(hp_type or "", {})
            if not any(stat in override for stat, _ in COLUMNS):
                continue
            variant = _variant_from_set(row)
            # The fixed encoder, pinned to this variant...
            fixed = _encode(self.dex, species=species, level=level, variants=(variant,))
            # ...and the encoder in its no-set-source state, which emits exactly the flat iv=31
            # value the pre-fix code emitted unconditionally. Same function, same inputs, so the
            # only thing separating them is whether the override is honoured.
            mutated = _encode(self.dex, species=species, level=level, variants=())
            for stat, slot in COLUMNS:
                if not info.base_stats.get(stat):
                    continue
                checked += 1
                if fixed[slot] != mutated[slot]:
                    killed += 1
        self.assertGreater(checked, 0, "mutation arm reached no override sets")
        self.assertGreater(
            killed,
            200,
            "the flat-iv31 mutation survived this differential -- the guard is not coverage",
        )


@requires_showdown()
class ExpectedStatEngineAnchorTest(unittest.TestCase):
    """The ENGINE arm, in CI -- short sweep of the same gate the fleet runs.

    Everything in ``ExpectedStatDifferentialTest`` compares the encoder against
    ``randbats_spread_details``. That is Python-vs-Python, and the plan's first standard says such
    a differential proves nothing on its own: both sides sharing the bug is how C1 survived. The
    engine anchor existed only inside ``scripts/expected_stat_gate.py``, so nothing in CI held the
    core to the engine. This closes that: server-computed stats from each seat's opening
    ``|request|``, compared against the core, the pinned encoder, and bound soundness.

    Deliberately few games -- it drives the real BattleStream. The 200-game run is the gate.
    """

    def test_short_engine_anchored_sweep_has_zero_mismatches(self) -> None:
        import sys

        scripts = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        if not (showdown_root() / "dist" / "sim" / "index.js").exists():
            self.skipTest("needs a BUILT Showdown checkout (dist/sim/index.js) and node")
        from expected_stat_gate import run_gate

        summary = run_gate(showdown_root=showdown_root(), games=2, seed=5)
        self.assertEqual(summary["core_mismatch_count"], 0, summary["core_mismatches"][:5])
        self.assertEqual(summary["pinned_mismatch_count"], 0, summary["pinned_mismatches"][:5])
        self.assertEqual(summary["bound_violation_count"], 0, summary["bound_violations"][:5])
        # Reachability: a sweep that compared nothing must not read as a pass.
        self.assertGreater(summary["counts"]["core_comparisons"], 0)
        self.assertGreater(summary["counts"]["pinned_comparisons"], 0)
        self.assertGreater(summary["counts"]["bound_comparisons"], 0)
        self.assertTrue(summary["reached_comparisons"])
        self.assertEqual(summary["verdict"], "PASS")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
