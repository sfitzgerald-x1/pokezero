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
   ``iv=31``, ignoring the generator's Hidden Power ``HPivs`` override, and were wrong by one
   point on 205 of 393 pool sets. ``test_flat_iv31_encoder_is_killed`` is the kill-confirmed
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


def _encode(dex, *, species: str, level: int, variants: tuple[Mapping[str, Any], ...]):
    """Run the real encoder block and return its four emitted column values."""
    num_row = [0.0] * _V4_NUMERIC_FEATURE_COUNT
    belief = RevealedPokemonBelief(
        showdown_slot="p2a",
        species=species,
        candidate_variants=variants,
    )
    _encode_expected_stats(
        num_row,
        dex,
        base_species=species,
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

    def test_ambiguous_candidates_emit_the_upper_bound_never_a_midpoint(self) -> None:
        """With candidates that disagree, the column is the MAX -- a real bound, not a fiction.

        30 of 220 pool species have candidate sets that disagree on at least one of the four
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

    def test_flat_iv31_encoder_is_killed(self) -> None:
        """Kill-confirmed mutation: the pre-fix encoder must FAIL this file's differential.

        The mutation is the exact code that shipped -- ``_gen3_stat(base, level, ev=85, iv=31)``
        with no Hidden Power override. If this test cannot tell the two apart, the differential
        above is not measuring anything.
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
            truth = self._truth(species, level, row)
            for stat, _slot in COLUMNS:
                base = info.base_stats.get(stat)
                if not base:
                    continue
                checked += 1
                mutated = _gen3_stat(int(base), level, ev=85, iv=31, hp=False)
                if mutated != truth[stat]:
                    killed += 1
        self.assertGreater(checked, 0, "mutation arm reached no override sets")
        self.assertGreater(
            killed,
            200,
            "the flat-iv31 mutation survived this differential -- the guard is not coverage",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
