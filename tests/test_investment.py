"""Defender-side investment inference tests (v2.1 batch 2).

Synthetic protocol lines + handcrafted dex payload + fake candidate sources, mirroring
tests/test_tier2.py. Numeric fixtures are computed through the same public helpers the
module uses (randbats_spread_details / gen3_damage_rolls) with setup assertions pinning
the structural properties each test relies on (family separations, roll membership).
"""

import unittest

from pokezero.belief import CandidateSetSummary, PublicBattleBeliefEngine
from pokezero.dex import showdown_dex_from_payload
from pokezero.gen3_damage import (
    Gen3DamageContext,
    gen3_damage_rolls,
    randbats_spread_details,
    randbats_spread_stats,
)
from pokezero.investment import (
    InvestmentConfig,
    InvestmentConclusion,
    InvestmentLiveTracker,
    _AxisLedger,
    conclusion_column_code,
    infer_investment,
)
from pokezero.showdown import parse_showdown_replay
from pokezero.tier2 import OwnMon
from pokezero.transitions import extract_transition_tokens


def _move_payload(move_id, name, move_type, category, base_power, accuracy=100):
    return {
        "id": move_id,
        "name": name,
        "type": move_type,
        "category": category,
        "basePower": base_power,
        "accuracy": accuracy,
        "priority": 0,
        "recoil": False,
        "drain": False,
        "heal": False,
        "status": None,
        "boosts": {},
        "target": "normal",
        "selfdestruct": False,
        "secondaries": [],
    }


def _species_payload(species_id, name, types, base_stats):
    return {"id": species_id, "name": name, "types": types, "baseStats": base_stats}


_DEX = showdown_dex_from_payload(
    {
        "moves": {
            "doubleedge": _move_payload("doubleedge", "Double-Edge", "Normal", "Physical", 120),
            "seismictoss": _move_payload("seismictoss", "Seismic Toss", "Fighting", "Physical", 0),
            "flamethrower": _move_payload("flamethrower", "Flamethrower", "Fire", "Special", 95),
            "hiddenpowerbug": _move_payload("hiddenpowerbug", "Hidden Power", "Bug", "Physical", 70),
            "surf": _move_payload("surf", "Surf", "Water", "Special", 95),
            "rest": _move_payload("rest", "Rest", "Psychic", "Status", 0),
            "sleeptalk": _move_payload("sleeptalk", "Sleep Talk", "Normal", "Status", 0),
            "bellydrum": _move_payload("bellydrum", "Belly Drum", "Normal", "Status", 0),
            "substitute": _move_payload("substitute", "Substitute", "Normal", "Status", 0),
        },
        "species": {
            "flareon": _species_payload(
                "flareon", "Flareon", ["Fire"],
                {"hp": 65, "atk": 130, "def": 60, "spa": 95, "spd": 110, "spe": 65},
            ),
            # Synthetic defender: base HP 94 puts the 85-EV L80 max HP at 282 (even), so
            # a Belly Drum variant trims to 281; base Def 110 puts IV-31 at 222 and the
            # Hidden-Power-Bug IV-30 override at 221.
            "slowbro": _species_payload(
                "slowbro", "Slowbro", ["Water", "Psychic"],
                {"hp": 94, "atk": 75, "def": 110, "spa": 100, "spd": 80, "spe": 30},
            ),
        },
        # damageTaken codes: 0 neutral, 1 weak (2x), 2 resist, 3 immune.
        "typeChart": {
            "normal": {"fighting": 1, "ghost": 3},
            "water": {"grass": 1, "fire": 2, "water": 2},
            "psychic": {"dark": 1, "psychic": 2},
            "fire": {"water": 1, "fire": 2, "grass": 2},
            "bug": {"fire": 1, "fighting": 2},
        },
    }
)

_LEVEL = 80

# Defender variant families. TRIMMED carries Belly Drum (max HP 281); FULL does not
# (282). Both carry Double-Edge so the Atk-zeroing rule stays out of the picture.
_VARIANT_TRIMMED = {
    "variant_id": "bro-trim",
    "moves": ["bellydrum", "doubleedge", "rest", "sleeptalk"],
    "ability": "Oblivious",
    "item": "Leftovers",
    "level": _LEVEL,
}
_VARIANT_FULL = {
    "variant_id": "bro-full",
    "moves": ["surf", "doubleedge", "rest", "sleeptalk"],
    "ability": "Oblivious",
    "item": "Leftovers",
    "level": _LEVEL,
}
# Defense family variants: Hidden Power Bug pins the Def IV at 30 (221); the plain
# variant stays IV 31 (222). Same max HP (Bug's overrides do not touch the HP IV).
_VARIANT_DEF_REDUCED = {
    "variant_id": "bro-hpbug",
    "moves": ["hiddenpowerbug", "surf", "rest", "sleeptalk"],
    "ability": "Oblivious",
    "item": "Leftovers",
    "level": _LEVEL,
}
_VARIANT_DEF_FULL = dict(_VARIANT_FULL)


class FakeSource:
    """PokemonSetSource returning fixed candidate variants per species."""

    def __init__(self, variants_by_species):
        self._variants = {key.lower(): list(value) for key, value in variants_by_species.items()}

    def summarize(self, *, format_id, species, revealed_moves, **kwargs):
        variants = self._variants.get("".join(ch for ch in species.lower() if ch.isalnum()))
        if variants is None:
            return None
        revealed = ["".join(ch for ch in str(m).lower() if ch.isalnum()) for m in revealed_moves]
        surviving = [
            variant
            for variant in variants
            if all(
                any(
                    move == reveal or (reveal == "hiddenpower" and move.startswith("hiddenpower"))
                    for move in variant["moves"]
                )
                for reveal in revealed
            )
        ]
        if not surviving:
            surviving = variants
        return CandidateSetSummary(
            species=species,
            candidate_count=len(surviving),
            uncertainty=len(surviving) / max(1, len(variants)),
            possible_abilities=tuple({str(v.get("ability") or "") for v in surviving}),
            possible_items=tuple({str(v.get("item") or "") for v in surviving}),
            possible_moves=tuple({m for v in surviving for m in v["moves"]}),
            candidate_variants=tuple(dict(v) for v in surviving),
        )


def _spread(variant):
    return randbats_spread_details(
        _DEX.species_info("slowbro").base_stats,
        level=variant["level"],
        moves=variant["moves"],
        item=variant["item"],
        has_physical_attack=True,
    )


_TRIMMED_HP = _spread(_VARIANT_TRIMMED).stats["hp"]  # 281
_FULL_HP = _spread(_VARIANT_FULL).stats["hp"]  # 282
_DEF_REDUCED = _spread(_VARIANT_DEF_REDUCED).stats["def"]  # 221
_DEF_FULL = _spread(_VARIANT_DEF_FULL).stats["def"]  # 222

_OWN_FLAREON = OwnMon(
    species="Flareon",
    level=_LEVEL,
    stats={"hp": 300, "atk": 350, "def": 150, "spa": 200, "spd": 210, "spe": 170},
    ability="Flash Fire",
    item="Leftovers",
    moves=("doubleedge", "seismictoss", "flamethrower"),
)
_OWN_TEAM = (_OWN_FLAREON,)


def _rolls(defense, *, crit=False):
    return gen3_damage_rolls(
        Gen3DamageContext(
            level=_LEVEL,
            base_power=120,
            category="Physical",
            attack=_OWN_FLAREON.stats["atk"],
            defense=defense,
            stab=False,
            effectiveness=1.0,
            crit=crit,
        )
    )


def _leads(defender_hp):
    return [
        "|player|p1|Alice|",
        "|player|p2|Bob|",
        "|switch|p1a: Flareon|Flareon, L80|300/300",
        f"|switch|p2a: Slowbro|Slowbro, L{_LEVEL}|{defender_hp}/{defender_hp}",
        "|turn|1",
    ]


def _strike_lines(damage_hp, *, turn, max_hp, prior_hp=None, move="Double-Edge",
                  extra_before=(), extra_after=()):
    prior = max_hp if prior_hp is None else prior_hp
    new_hp = prior - damage_hp
    condition = f"{new_hp}/{max_hp}" if new_hp > 0 else "0 fnt"
    lines = [f"|move|p1a: Flareon|{move}|p2a: Slowbro"]
    lines.extend(extra_before)
    lines.append(f"|-damage|p2a: Slowbro|{condition}")
    lines.extend(extra_after)
    lines.extend(["|", "|upkeep", f"|turn|{turn + 1}"])
    return lines


def _infer(lines, *, source=None, config=None, own_team=_OWN_TEAM):
    replay = parse_showdown_replay(lines)
    return infer_investment(
        replay,
        perspective_slot="p1",
        own_team=own_team,
        dex=_DEX,
        set_source=source
        if source is not None
        else FakeSource({"slowbro": [_VARIANT_TRIMMED, _VARIANT_FULL]}),
        config=config or InvestmentConfig(),
    )


class SpreadDetailsTest(unittest.TestCase):
    def test_details_expose_the_trim_and_match_stats(self) -> None:
        trimmed = _spread(_VARIANT_TRIMMED)
        full = _spread(_VARIANT_FULL)
        self.assertEqual(full.evs["hp"], 85)
        self.assertLess(trimmed.evs["hp"], 85)
        self.assertEqual(full.stats["hp"], _FULL_HP)
        self.assertEqual(trimmed.stats["hp"], _TRIMMED_HP)
        self.assertEqual(_FULL_HP - _TRIMMED_HP, 1)
        stats_only = randbats_spread_stats(
            _DEX.species_info("slowbro").base_stats,
            level=_LEVEL,
            moves=_VARIANT_TRIMMED["moves"],
            item=_VARIANT_TRIMMED["item"],
            has_physical_attack=True,
        )
        self.assertEqual(stats_only, dict(trimmed.stats))

    def test_setup_family_separations(self) -> None:
        # The structural properties every test below relies on.
        self.assertEqual(_DEF_FULL - _DEF_REDUCED, 1)
        rolls = _rolls(_DEF_FULL)
        self.assertIn(130, rolls)
        # Max-roll strike on the trimmed denominator: the full family misses by ~0.46.
        shift = 130.0 * _FULL_HP / _TRIMMED_HP - 130.0
        self.assertGreater(shift, 0.25 + 1e-6)
        self.assertLess(shift, 1.0 - (0.25 + 1e-6))
        # Def families produce different lattices with a unique max roll.
        self.assertNotEqual(_rolls(_DEF_REDUCED), _rolls(_DEF_FULL))
        self.assertIn(131, set(_rolls(_DEF_REDUCED)) - set(_rolls(_DEF_FULL)))


class HpPinTest(unittest.TestCase):
    def test_two_strikes_pin_trimmed_hp(self) -> None:
        damage = 130
        lines = _leads(_TRIMMED_HP)
        lines += _strike_lines(damage, turn=1, max_hp=_TRIMMED_HP)
        lines += _strike_lines(damage, turn=2, max_hp=_TRIMMED_HP, prior_hp=_TRIMMED_HP - damage)
        inference = _infer(lines)
        strikes = inference.strikes
        self.assertEqual(len(strikes), 2)
        for strike in strikes:
            self.assertEqual(strike.disqualifiers, ())
            self.assertEqual(strike.hp_pin, _TRIMMED_HP)
            self.assertEqual(strike.hp_pin_class, "trimmed")
            self.assertEqual(strike.candidate_hp_values, (_TRIMMED_HP, _FULL_HP))
        conclusion = inference.conclusions["p2:slowbro"]
        self.assertEqual(conclusion.hp_value, _TRIMMED_HP)
        self.assertEqual(conclusion.hp_class, "trimmed")
        self.assertEqual(conclusion.hp_pin_turns, (1, 2))
        # Tokens: two lead switches, then the two strike tokens. The code lands on the
        # CONCLUDING strike's token only (two-strike rule), value -1 (trimmed).
        self.assertEqual(inference.token_codes, {3: -1.0})

    def test_two_strikes_pin_full_hp(self) -> None:
        damage = 130
        lines = _leads(_FULL_HP)
        lines += _strike_lines(damage, turn=1, max_hp=_FULL_HP)
        lines += _strike_lines(damage, turn=2, max_hp=_FULL_HP, prior_hp=_FULL_HP - damage)
        inference = _infer(lines)
        conclusion = inference.conclusions["p2:slowbro"]
        self.assertEqual(conclusion.hp_value, _FULL_HP)
        self.assertEqual(conclusion.hp_class, "full")
        self.assertEqual(inference.token_codes, {3: 1.0})

    def test_single_strike_never_concludes(self) -> None:
        lines = _leads(_TRIMMED_HP)
        lines += _strike_lines(130, turn=1, max_hp=_TRIMMED_HP)
        inference = _infer(lines)
        self.assertEqual(inference.strikes[0].hp_pin, _TRIMMED_HP)
        conclusion = inference.conclusions["p2:slowbro"]
        self.assertIsNone(conclusion.hp_value)
        self.assertEqual(inference.token_codes, {})

    def test_seismic_toss_pins_the_denominator(self) -> None:
        # Fixed exact damage (our level, 80): the trimmed family matches exactly, the
        # full family misses by 80/281 which must clear the rejection margin.
        shift = 80.0 * _FULL_HP / _TRIMMED_HP - 80.0
        self.assertGreater(shift, 0.25 + 1e-6)
        lines = _leads(_TRIMMED_HP)
        lines += _strike_lines(80, turn=1, max_hp=_TRIMMED_HP, move="Seismic Toss")
        lines += _strike_lines(80, turn=2, max_hp=_TRIMMED_HP, prior_hp=_TRIMMED_HP - 80,
                               move="Seismic Toss")
        inference = _infer(lines)
        for strike in inference.strikes:
            self.assertEqual(strike.disqualifiers, ())
            self.assertEqual(strike.hp_pin, _TRIMMED_HP)
            self.assertIsNone(strike.defense_stat_key)  # constants probe no defense stat
        conclusion = inference.conclusions["p2:slowbro"]
        self.assertEqual(conclusion.hp_value, _TRIMMED_HP)
        self.assertEqual(conclusion.defense_values, {})

    def test_crit_strike_pins_with_crit_conditioned_rolls(self) -> None:
        # A weak attacker keeps two crit strikes below the KO line; the observed
        # damage must match the CRIT-conditioned lattice (the plain rolls miss it).
        weak_attack = 100
        weak = OwnMon(
            species="Flareon", level=_LEVEL,
            stats={"hp": 300, "atk": weak_attack, "def": 150, "spa": 200, "spd": 210, "spe": 170},
            ability="Flash Fire", item="Leftovers",
            moves=("doubleedge", "seismictoss"),
        )

        def rolls(crit):
            return gen3_damage_rolls(
                Gen3DamageContext(
                    level=_LEVEL, base_power=120, category="Physical",
                    attack=weak_attack, defense=_DEF_FULL, stab=False,
                    effectiveness=1.0, crit=crit,
                )
            )

        damage = max(rolls(crit=True))
        self.assertNotIn(damage, rolls(crit=False))
        self.assertLess(2 * damage, _TRIMMED_HP)
        shift = damage * float(_FULL_HP) / _TRIMMED_HP - damage
        self.assertGreater(min(abs(damage + shift - r) for r in rolls(crit=True)), 0.25 + 1e-6)
        lines = _leads(_TRIMMED_HP)
        lines += _strike_lines(damage, turn=1, max_hp=_TRIMMED_HP,
                               extra_before=("|-crit|p2a: Slowbro",))
        lines += _strike_lines(damage, turn=2, max_hp=_TRIMMED_HP,
                               prior_hp=_TRIMMED_HP - damage,
                               extra_before=("|-crit|p2a: Slowbro",))
        inference = _infer(lines, own_team=(weak,))
        for strike in inference.strikes:
            self.assertEqual(strike.disqualifiers, ())
            self.assertEqual(strike.hp_pin, _TRIMMED_HP)
        self.assertEqual(inference.conclusions["p2:slowbro"].hp_value, _TRIMMED_HP)


class PrecisionGuardTest(unittest.TestCase):
    def test_ko_and_truncated_strikes_are_disqualified(self) -> None:
        lines = _leads(_TRIMMED_HP)
        lines += _strike_lines(_TRIMMED_HP, turn=1, max_hp=_TRIMMED_HP,
                               extra_after=("|faint|p2a: Slowbro",))
        inference = _infer(lines)
        self.assertIn("ko-clipped", inference.strikes[0].disqualifiers)

        sub_lines = _leads(_TRIMMED_HP)
        sub_lines += [
            "|move|p1a: Flareon|Double-Edge|p2a: Slowbro",
            "|-activate|p2a: Slowbro|Substitute|[damage]",
            "|", "|upkeep", "|turn|2",
        ]
        inference = _infer(sub_lines)
        self.assertEqual(inference.strikes[0].hp_pin, None)
        self.assertIn("no-damage-event", inference.strikes[0].disqualifiers)

    def test_off_model_observation_yields_no_evidence(self) -> None:
        # Candidate universe only contains the FULL variant, but the true denominator
        # is the trimmed 281: nothing is consistent — off-model, no pin, no rejection
        # bookkeeping (and no crash).
        source = FakeSource({"slowbro": [_VARIANT_FULL]})
        lines = _leads(_TRIMMED_HP)
        lines += _strike_lines(130, turn=1, max_hp=_TRIMMED_HP)
        lines += _strike_lines(130, turn=2, max_hp=_TRIMMED_HP, prior_hp=_TRIMMED_HP - 130)
        inference = _infer(lines, source=source)
        for strike in inference.strikes:
            self.assertTrue(strike.off_model)
            self.assertIsNone(strike.hp_pin)
        self.assertEqual(inference.conclusions, {})

    def test_single_family_universe_never_pins(self) -> None:
        # Belief-elimination vacuity: with one candidate family the strike is
        # consistent but pins nothing (mirrors cb-pinned-by-elimination).
        source = FakeSource({"slowbro": [_VARIANT_TRIMMED]})
        lines = _leads(_TRIMMED_HP)
        lines += _strike_lines(130, turn=1, max_hp=_TRIMMED_HP)
        lines += _strike_lines(130, turn=2, max_hp=_TRIMMED_HP, prior_hp=_TRIMMED_HP - 130)
        inference = _infer(lines, source=source)
        for strike in inference.strikes:
            self.assertEqual(strike.disqualifiers, ())
            self.assertIsNone(strike.hp_pin)
            self.assertEqual(strike.candidate_hp_values, (_TRIMMED_HP,))
        self.assertIsNone(inference.conclusions["p2:slowbro"].hp_value)

    def test_margin_band_blocks_the_pin(self) -> None:
        # A weak attacker makes the family shift (damage/281) smaller than the
        # rejection margin: the wrong family is neither consistent nor rejected, so
        # the strike must yield no pin.
        weak = OwnMon(
            species="Flareon", level=_LEVEL,
            stats={"hp": 300, "atk": 100, "def": 150, "spa": 200, "spd": 210, "spe": 170},
            ability="Flash Fire", item="Leftovers",
            moves=("doubleedge", "seismictoss"),
        )
        rolls = gen3_damage_rolls(
            Gen3DamageContext(
                level=_LEVEL, base_power=120, category="Physical",
                attack=100, defense=_DEF_FULL, stab=False, effectiveness=1.0,
            )
        )
        damage = max(rolls)
        shift = damage * float(_FULL_HP) / _TRIMMED_HP - damage
        self.assertLess(shift, 0.25)
        self.assertGreater(shift, 1e-6)
        lines = _leads(_TRIMMED_HP)
        lines += _strike_lines(damage, turn=1, max_hp=_TRIMMED_HP)
        inference = _infer(lines, own_team=(weak,))
        strike = inference.strikes[0]
        self.assertTrue(strike.margin_ambiguous)
        self.assertIn("margin-ambiguity", strike.disqualifiers)
        self.assertIsNone(strike.hp_pin)

    def test_percent_granularity_disarms_the_lattice(self) -> None:
        # Player-view quantization (1% of max HP) makes both families consistent:
        # conclusions must never fire on percent-quantized inputs.
        config = InvestmentConfig(fraction_granularity=0.01)
        lines = _leads(_TRIMMED_HP)
        lines += _strike_lines(130, turn=1, max_hp=_TRIMMED_HP)
        lines += _strike_lines(130, turn=2, max_hp=_TRIMMED_HP, prior_hp=_TRIMMED_HP - 130)
        inference = _infer(lines, config=config)
        for strike in inference.strikes:
            self.assertIsNone(strike.hp_pin)
            self.assertEqual(strike.consistent_hp_values, (_TRIMMED_HP, _FULL_HP))
        self.assertEqual(inference.conclusions["p2:slowbro"].hp_value, None)

    def test_axis_ledger_conflict_and_freeze_semantics(self) -> None:
        ledger = _AxisLedger()
        ledger.observe(pin_value=281, pin_class="trimmed", rejected_values=(282,), turn=1, required=2)
        # A conflicting pin before conclusion blocks the axis permanently.
        ledger.observe(pin_value=282, pin_class="full", rejected_values=(281,), turn=2, required=2)
        self.assertTrue(ledger.blocked)
        ledger.observe(pin_value=281, pin_class="trimmed", rejected_values=(282,), turn=3, required=2)
        self.assertIsNone(ledger.concluded_value)

        # Margin-rejection of a previously pinned value blocks even without a new pin.
        ledger = _AxisLedger()
        ledger.observe(pin_value=281, pin_class="trimmed", rejected_values=(282,), turn=1, required=2)
        ledger.observe(pin_value=None, pin_class=None, rejected_values=(281,), turn=2, required=2)
        self.assertTrue(ledger.blocked)

        # Conclusions freeze: post-conclusion observations change nothing.
        ledger = _AxisLedger()
        ledger.observe(pin_value=281, pin_class="trimmed", rejected_values=(282,), turn=1, required=2)
        ledger.observe(pin_value=281, pin_class="trimmed", rejected_values=(282,), turn=2, required=2)
        self.assertEqual(ledger.concluded_value, 281)
        ledger.observe(pin_value=282, pin_class="full", rejected_values=(281,), turn=3, required=2)
        self.assertEqual(ledger.concluded_value, 281)
        self.assertFalse(ledger.blocked)


class DefensePinTest(unittest.TestCase):
    def test_two_strikes_pin_reduced_def(self) -> None:
        source = FakeSource({"slowbro": [_VARIANT_DEF_REDUCED, _VARIANT_DEF_FULL]})
        damage = 131  # unique to the IV-30 def lattice (max roll)
        lines = _leads(_FULL_HP)
        lines += _strike_lines(damage, turn=1, max_hp=_FULL_HP)
        lines += _strike_lines(damage, turn=2, max_hp=_FULL_HP, prior_hp=_FULL_HP - damage)
        inference = _infer(lines, source=source)
        for strike in inference.strikes:
            self.assertEqual(strike.disqualifiers, ())
            self.assertEqual(strike.defense_pin, _DEF_REDUCED)
            self.assertIsNone(strike.hp_pin)  # both variants share max HP 282
            self.assertEqual(strike.candidate_hp_values, (_FULL_HP,))
        conclusion = inference.conclusions["p2:slowbro"]
        self.assertIsNone(conclusion.hp_value)
        self.assertEqual(conclusion.defense_values, {"def": _DEF_REDUCED})
        self.assertEqual(conclusion.defense_classes, {"def": "reduced"})
        self.assertEqual(inference.token_codes, {3: -0.5})

    def test_two_strikes_pin_full_def(self) -> None:
        source = FakeSource({"slowbro": [_VARIANT_DEF_REDUCED, _VARIANT_DEF_FULL]})
        reduced = set(_rolls(_DEF_REDUCED))
        damage = max(set(_rolls(_DEF_FULL)) - reduced,
                     key=lambda r: min(abs(r - other) for other in reduced))
        self.assertGreaterEqual(min(abs(damage - r) for r in reduced), 1)
        lines = _leads(_FULL_HP)
        lines += _strike_lines(damage, turn=1, max_hp=_FULL_HP)
        lines += _strike_lines(damage, turn=2, max_hp=_FULL_HP, prior_hp=_FULL_HP - damage)
        inference = _infer(lines, source=source)
        conclusion = inference.conclusions["p2:slowbro"]
        self.assertEqual(conclusion.defense_values, {"def": _DEF_FULL})
        self.assertEqual(conclusion.defense_classes, {"def": "full"})
        self.assertEqual(inference.token_codes, {3: 0.5})


class ColumnCodeTest(unittest.TestCase):
    def test_projection_precedence(self) -> None:
        self.assertEqual(
            conclusion_column_code(InvestmentConclusion("k", hp_value=281, hp_class="trimmed")), -1.0
        )
        self.assertEqual(
            conclusion_column_code(InvestmentConclusion("k", hp_value=282, hp_class="full")), 1.0
        )
        # Value-only HP pin (mixed classes) does not encode.
        self.assertEqual(conclusion_column_code(InvestmentConclusion("k", hp_value=282)), 0.0)
        self.assertEqual(
            conclusion_column_code(
                InvestmentConclusion("k", defense_values={"def": 221}, defense_classes={"def": "reduced"})
            ),
            -0.5,
        )
        self.assertEqual(
            conclusion_column_code(
                InvestmentConclusion("k", defense_values={"spd": 222}, defense_classes={"spd": "full"})
            ),
            0.5,
        )
        # HP dominates defense.
        self.assertEqual(
            conclusion_column_code(
                InvestmentConclusion(
                    "k", hp_value=282, hp_class="full",
                    defense_values={"def": 221}, defense_classes={"def": "reduced"},
                )
            ),
            1.0,
        )
        self.assertEqual(conclusion_column_code(InvestmentConclusion("k")), 0.0)


class LiveTrackerTest(unittest.TestCase):
    def test_incremental_matches_batch(self) -> None:
        damage = 130
        lines = _leads(_TRIMMED_HP)
        lines += _strike_lines(damage, turn=1, max_hp=_TRIMMED_HP)
        lines += _strike_lines(damage, turn=2, max_hp=_TRIMMED_HP, prior_hp=_TRIMMED_HP - damage)
        batch = _infer(lines)

        source = FakeSource({"slowbro": [_VARIANT_TRIMMED, _VARIANT_FULL]})
        engine = PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=source)
        tracker = InvestmentLiveTracker(
            perspective_slot="p1", own_team=_OWN_TEAM, dex=_DEX
        )
        fed = 0
        codes: dict[int, float] = {}
        # Feed in three chunks at complete window boundaries (after leads, after the
        # strike-1 chunk, end of log) — live consumers observe at request boundaries,
        # never mid-window.
        for upto in (5, 10, len(lines)):
            replay = parse_showdown_replay(lines[:upto])
            while fed < len(replay.public_events):
                engine.ingest_event(replay.public_events[fed])
                fed += 1
            tokens = extract_transition_tokens(replay, perspective_slot="p1")
            codes = tracker.observe(replay, tokens, engine)
        self.assertEqual(codes, dict(batch.token_codes))
        self.assertEqual(
            tracker.conclusions["p2:slowbro"].hp_value,
            batch.conclusions["p2:slowbro"].hp_value,
        )
        self.assertEqual(tracker.conclusions["p2:slowbro"].hp_class, "trimmed")

    def test_clone_isolates_nonempty_inference_state(self) -> None:
        damage = 130
        lines = _leads(_TRIMMED_HP)
        lines += _strike_lines(damage, turn=1, max_hp=_TRIMMED_HP)
        lines += _strike_lines(damage, turn=2, max_hp=_TRIMMED_HP, prior_hp=_TRIMMED_HP - damage)

        source = FakeSource({"slowbro": [_VARIANT_TRIMMED, _VARIANT_FULL]})
        engine = PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=source)
        tracker = InvestmentLiveTracker(
            perspective_slot="p1",
            own_team=_OWN_TEAM,
            dex=_DEX,
        )
        fed = 0
        for upto in (5, 10, len(lines)):
            replay = parse_showdown_replay(lines[:upto])
            while fed < len(replay.public_events):
                engine.ingest_event(replay.public_events[fed])
                fed += 1
            tracker.observe(replay, extract_transition_tokens(replay, perspective_slot="p1"), engine)

        self.assertTrue(tracker._state.ledgers)
        self.assertTrue(tracker._state.token_codes)
        original_codes = dict(tracker._state.token_codes)
        original_strikes = list(tracker._state.strikes)
        original_levels = dict(tracker._defender_levels)
        defender_key = next(iter(tracker._state.ledgers))
        original_hp_pins = list(tracker._state.ledgers[defender_key].hp.pins)

        cloned = tracker.clone()
        cloned._state.token_codes[-1] = -1.0
        cloned._state.strikes.pop()
        cloned._state.ledgers[defender_key].hp.pins.append((99, 999, "clone-only"))
        cloned._defender_levels["p2:clone-only"] = 1
        cloned._fold.boosts["p1"]["atk"] = 6

        self.assertEqual(tracker._state.token_codes, original_codes)
        self.assertEqual(tracker._state.strikes, original_strikes)
        self.assertEqual(tracker._defender_levels, original_levels)
        self.assertEqual(tracker._state.ledgers[defender_key].hp.pins, original_hp_pins)
        self.assertNotIn("atk", tracker._fold.boosts["p1"])


if __name__ == "__main__":
    unittest.main()
