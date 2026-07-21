"""Tier-2 residual + Choice Band inference tests (PR D).

All tests run on synthetic protocol lines with a handcrafted dex payload and fake
candidate-set sources — no node / Showdown checkout required. The damage arithmetic
itself is validated separately (tests/test_gen3_damage.py, incl. live-sim cross-checks);
these tests validate the CONDITIONING and INFERENCE plumbing by comparing infer_tier2's
outputs against directly-constructed Gen3DamageContext computations.
"""

import unittest
from types import SimpleNamespace

from pokezero.belief import CandidateSetSummary
from pokezero.dex import showdown_dex_from_payload
from pokezero.gen3_damage import (
    Gen3DamageContext,
    gen3_damage_rolls,
    median_damage,
    randbats_spread_stats,
)
from pokezero.showdown import parse_showdown_replay
from pokezero.tier2 import (
    OwnMon,
    Tier2Config,
    Tier2LiveTracker,
    _IncrementalContextFold,
    apply_residuals,
    build_cb_whitelist,
    infer_tier2,
    own_team_from_request,
    variant_has_physical_attack,
)
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
        "selfdestruct": move_id in {"explosion", "selfdestruct"},
        "secondaries": [],
    }


def _species_payload(species_id, name, types, base_stats):
    return {"id": species_id, "name": name, "types": types, "baseStats": base_stats}


_DEX = showdown_dex_from_payload(
    {
        "moves": {
            "bodyslam": _move_payload("bodyslam", "Body Slam", "Normal", "Physical", 85),
            "brickbreak": _move_payload("brickbreak", "Brick Break", "Fighting", "Physical", 75),
            "return": _move_payload("return", "Return", "Normal", "Physical", 0),
            "earthquake": _move_payload("earthquake", "Earthquake", "Ground", "Physical", 100),
            "rockslide": _move_payload("rockslide", "Rock Slide", "Rock", "Physical", 75, accuracy=90),
            "doubleedge": _move_payload("doubleedge", "Double-Edge", "Normal", "Physical", 120),
            "flamethrower": _move_payload("flamethrower", "Flamethrower", "Fire", "Special", 95),
            "solarbeam": _move_payload("solarbeam", "Solar Beam", "Grass", "Special", 120),
            "facade": _move_payload("facade", "Facade", "Normal", "Physical", 70),
            "bonemerang": _move_payload("bonemerang", "Bonemerang", "Ground", "Physical", 60),
            "flail": _move_payload("flail", "Flail", "Normal", "Physical", 0),
            "pursuit": _move_payload("pursuit", "Pursuit", "Dark", "Special", 40),
            "counter": _move_payload("counter", "Counter", "Fighting", "Physical", 0),
            "hiddenpowergrass": _move_payload("hiddenpowergrass", "Hidden Power", "Grass", "Special", 70),
            "rest": _move_payload("rest", "Rest", "Psychic", "Status", 0),
            "sleeptalk": _move_payload("sleeptalk", "Sleep Talk", "Normal", "Status", 0),
        },
        "species": {
            "snorlax": _species_payload(
                "snorlax", "Snorlax", ["Normal"],
                {"hp": 160, "atk": 110, "def": 65, "spa": 65, "spd": 110, "spe": 30},
            ),
            "tyranitar": _species_payload(
                "tyranitar", "Tyranitar", ["Rock", "Dark"],
                {"hp": 100, "atk": 134, "def": 110, "spa": 95, "spd": 100, "spe": 61},
            ),
            "charizard": _species_payload(
                "charizard", "Charizard", ["Fire", "Flying"],
                {"hp": 78, "atk": 84, "def": 78, "spa": 109, "spd": 85, "spe": 100},
            ),
            "slowbro": _species_payload(
                "slowbro", "Slowbro", ["Water", "Psychic"],
                {"hp": 95, "atk": 75, "def": 110, "spa": 100, "spd": 80, "spe": 30},
            ),
            "flareon": _species_payload(
                "flareon", "Flareon", ["Fire"],
                {"hp": 65, "atk": 130, "def": 60, "spa": 95, "spd": 110, "spe": 65},
            ),
        },
        # damageTaken codes: 0 neutral, 1 weak (2x), 2 resist, 3 immune.
        "typeChart": {
            "normal": {"fighting": 1, "ghost": 3},
            "water": {"grass": 1, "fire": 2, "water": 2},
            "psychic": {"dark": 1, "psychic": 2},
            "fire": {"water": 1, "fire": 2, "grass": 2},
            "flying": {"grass": 2, "ground": 3},
            "rock": {"grass": 1, "fire": 2, "normal": 2},
            "dark": {"psychic": 3},
            "grass": {"fire": 1, "water": 2},
            "ground": {"water": 1, "grass": 1, "electric": 3},
        },
    }
)


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


_LEVEL = 80

# Snorlax attacker candidates: same moves, Leftovers vs Choice Band.
_SNORLAX_MOVES = ["bodyslam", "earthquake", "rest", "sleeptalk"]
_SNORLAX_VARIANTS = [
    {"variant_id": "lax-1", "moves": _SNORLAX_MOVES, "ability": "Immunity", "item": "Leftovers", "level": _LEVEL},
    {"variant_id": "lax-2", "moves": _SNORLAX_MOVES, "ability": "Immunity", "item": "Choice Band", "level": _LEVEL},
]

# The perspective player's own mon (defender): exact stats.
_OWN_SLOWBRO = OwnMon(
    species="Slowbro",
    level=80,
    stats={"hp": 330, "atk": 150, "def": 230, "spa": 210, "spd": 180, "spe": 100},
    ability="Oblivious",
    item="Leftovers",
)
_OWN_TEAM = (_OWN_SLOWBRO,)


def _snorlax_stats():
    return randbats_spread_stats(
        _DEX.species_info("snorlax").base_stats,
        level=_LEVEL,
        moves=_SNORLAX_MOVES,
        item="Leftovers",
        has_physical_attack=True,
    )


def _bodyslam_rolls(*, cb=False, crit=False, screen=False, attack_boost=0, defense_boost=0, burned=False):
    stats = _snorlax_stats()
    return gen3_damage_rolls(
        Gen3DamageContext(
            level=_LEVEL,
            base_power=85,
            category="Physical",
            attack=stats["atk"],
            defense=_OWN_SLOWBRO.stats["def"],
            attack_mods=((1.5, 1),) if cb else (),
            stab=True,
            effectiveness=1.0,
            crit=crit,
            screen=screen,
            attack_boost=attack_boost,
            defense_boost=defense_boost,
            burned=burned,
        )
    )


_WHITELIST = {"snorlax": frozenset({"bodyslam", "earthquake"})}


def _leads(p1="Slowbro", p2="Snorlax", p1_hp="330/330", p2_hp="100/100"):
    return [
        "|player|p1|Alice|",
        "|player|p2|Bob|",
        f"|switch|p1a: {p1}|{p1}, L80|{p1_hp}",
        f"|switch|p2a: {p2}|{p2}, L{_LEVEL}|{p2_hp}",
        "|turn|1",
    ]


def _strike_lines(damage_hp, *, turn, prior_hp=330, move="Body Slam", extra_before=(), extra_after=()):
    new_hp = prior_hp - damage_hp
    condition = f"{new_hp}/330" if new_hp > 0 else "0 fnt"
    lines = [f"|move|p2a: Snorlax|{move}|p1a: Slowbro"]
    lines.extend(extra_before)
    lines.append(f"|-damage|p1a: Slowbro|{condition}")
    lines.extend(extra_after)
    lines.extend(["|", "|upkeep", f"|turn|{turn + 1}"])
    return lines


def _infer(lines, *, source=None, whitelist=None, config=None, own_team=_OWN_TEAM):
    replay = parse_showdown_replay(lines)
    return infer_tier2(
        replay,
        perspective_slot="p1",
        own_team=own_team,
        dex=_DEX,
        set_source=source if source is not None else FakeSource({"snorlax": _SNORLAX_VARIANTS}),
        whitelist=whitelist if whitelist is not None else _WHITELIST,
        config=config or Tier2Config(),
    )


def _exceeding_damage():
    """A damage value that exceeds the non-CB max by more than the margin but stays
    within the CB explanation."""
    non_cb_max = max(_bodyslam_rolls(cb=False))
    cb_max = max(_bodyslam_rolls(cb=True))
    margin = 0.01 * 330 + 1.0
    observed = int(non_cb_max + margin + 2)
    assert observed <= cb_max, "test setup: exceedance must stay CB-explainable"
    return observed


class ChoiceBandTwoStrikeTest(unittest.TestCase):
    def test_two_clean_exceedances_set_the_bit(self) -> None:
        damage = _exceeding_damage()
        lines = _leads()
        lines += _strike_lines(damage, turn=1)
        lines += _strike_lines(damage, turn=2, prior_hp=330 - damage)
        inference = _infer(lines)
        self.assertEqual(inference.cb_bits.get("p2:snorlax"), True)
        self.assertEqual(len(inference.cb_strike_turns["p2:snorlax"]), 2)
        # The as-of-strike CB bit (slot 119's source field): False on the first
        # exceedance token (conclusion needs two), True from the second onward.
        strike_tokens = [t for t in inference.tokens if t.kind == "move" and t.actor_slot == "p2"]
        self.assertEqual([t.cb_bit for t in strike_tokens], [False, True])

    def test_single_exceedance_is_not_enough(self) -> None:
        damage = _exceeding_damage()
        lines = _leads() + _strike_lines(damage, turn=1)
        inference = _infer(lines)
        # One exceedance is tracked but the bit stays off (two-strike rule).
        self.assertFalse(inference.cb_bits.get("p2:snorlax", False))
        self.assertEqual(len(inference.cb_strike_turns["p2:snorlax"]), 1)

    def test_non_exceeding_damage_never_counts(self) -> None:
        damage = int(median_damage(_bodyslam_rolls(cb=False)))
        lines = _leads()
        lines += _strike_lines(damage, turn=1)
        lines += _strike_lines(damage, turn=2, prior_hp=330 - damage)
        inference = _infer(lines)
        self.assertNotIn("p2:snorlax", inference.cb_bits)
        # The residual channel still populated (clean strike, agreeing baseline).
        strikes = [s for s in inference.strikes if s.residual_valid]
        self.assertEqual(len(strikes), 2)

    def test_crit_strikes_never_count_but_residual_conditions_on_crit(self) -> None:
        crit_damage = int(median_damage(_bodyslam_rolls(cb=False, crit=True)))
        lines = _leads()
        lines += _strike_lines(
            crit_damage, turn=1, extra_before=["|-crit|p1a: Slowbro"]
        )
        lines += _strike_lines(
            crit_damage, turn=2, prior_hp=330 - crit_damage, extra_before=["|-crit|p1a: Slowbro"]
        )
        inference = _infer(lines)
        self.assertNotIn("p2:snorlax", inference.cb_bits)
        strike = inference.strikes[0]
        self.assertIn("crit", strike.disqualifiers)
        self.assertTrue(strike.residual_valid)
        expected = median_damage(_bodyslam_rolls(cb=False, crit=True))
        self.assertAlmostEqual(strike.expected_median_hp, expected)

    def test_screen_disqualifies(self) -> None:
        damage = _exceeding_damage()
        lines = _leads()
        lines += ["|move|p1a: Slowbro|Reflect|p1a: Slowbro", "|-sidestart|p1: Alice|Reflect"]
        lines += _strike_lines(damage, turn=1)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertIn("screen", strike.disqualifiers)
        self.assertFalse(strike.cb_eligible)
        # The residual still conditions on the screen deterministically.
        self.assertTrue(strike.residual_valid)
        self.assertAlmostEqual(
            strike.expected_median_hp, median_damage(_bodyslam_rolls(screen=True))
        )

    def test_brick_break_conditions_unscreened_through_reflect(self) -> None:
        # Brick Break shatters Reflect in its onTryHit BEFORE dealing damage, so its own
        # strike lands unscreened even though Reflect is live at the |move| line. The
        # residual must condition on the UNSCREENED lattice (the observed high roll is
        # impossible under the screened one), and the strike must NOT be screen-disqualified.
        stats = _snorlax_stats()

        def bb_rolls(screen):
            return gen3_damage_rolls(
                Gen3DamageContext(
                    level=_LEVEL, base_power=75, category="Physical",
                    attack=stats["atk"], defense=_OWN_SLOWBRO.stats["def"],
                    stab=False, effectiveness=1.0, screen=screen,
                )
            )

        unscreened, screened = bb_rolls(False), bb_rolls(True)
        observed = max(unscreened)
        self.assertNotIn(observed, screened, "fixture: high unscreened roll must miss the screened lattice")
        source = FakeSource(
            {"snorlax": [{"variant_id": "bb", "moves": ["brickbreak", "earthquake", "rest", "sleeptalk"],
                          "ability": "Immunity", "item": "Leftovers", "level": _LEVEL}]}
        )
        lines = _leads()
        lines += ["|move|p1a: Slowbro|Reflect|p1a: Slowbro", "|-sidestart|p1: Alice|Reflect"]
        # Engine order: the shatter -sideend precedes the -damage of this same strike.
        lines += _strike_lines(observed, turn=1, move="Brick Break",
                               extra_before=["|-sideend|p1: Alice|Reflect"])
        inference = _infer(lines, source=source, whitelist={"snorlax": frozenset({"brickbreak"})})
        strike = next(s for s in inference.strikes if s.move_id == "brickbreak")
        self.assertNotIn("screen", strike.disqualifiers)
        self.assertTrue(strike.residual_valid)
        self.assertAlmostEqual(strike.expected_median_hp, median_damage(unscreened))
        self.assertNotAlmostEqual(strike.expected_median_hp, median_damage(screened))
        self.assertGreater(strike.residual, 0)
        self.assertAlmostEqual(strike.residual, (observed - median_damage(unscreened)) / 330.0)

    def test_stat_stages_disqualify_cb_but_condition_residual(self) -> None:
        damage = _exceeding_damage()
        lines = _leads()
        lines += ["|-boost|p2a: Snorlax|atk|1"]
        lines += _strike_lines(damage, turn=1)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertIn("stat-stages", strike.disqualifiers)
        self.assertFalse(strike.cb_eligible)
        self.assertAlmostEqual(
            strike.expected_median_hp, median_damage(_bodyslam_rolls(attack_boost=1))
        )

    def test_endured_truncation_disqualifies_everything(self) -> None:
        lines = _leads()
        lines += [
            "|move|p2a: Snorlax|Body Slam|p1a: Slowbro",
            "|-activate|p1a: Slowbro|move: Endure",
            "|-damage|p1a: Slowbro|1/330",
            "|",
            "|upkeep",
            "|turn|2",
        ]
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertIn("truncated-outcome", strike.disqualifiers)
        self.assertFalse(strike.residual_valid)
        self.assertFalse(strike.cb_eligible)

    def test_cb_not_a_candidate_never_fires(self) -> None:
        source = FakeSource(
            {"snorlax": [{"variant_id": "lax-1", "moves": _SNORLAX_MOVES, "ability": "Immunity", "item": "Leftovers", "level": _LEVEL}]}
        )
        damage = _exceeding_damage()
        lines = _leads()
        lines += _strike_lines(damage, turn=1)
        lines += _strike_lines(damage, turn=2, prior_hp=330 - damage)
        inference = _infer(lines, source=source)
        self.assertNotIn("p2:snorlax", inference.cb_bits)
        strike = inference.strikes[0]
        self.assertIn("cb-not-a-candidate", strike.disqualifiers)

    def test_cb_pinned_by_elimination_is_left_to_tier1(self) -> None:
        source = FakeSource(
            {"snorlax": [{"variant_id": "lax-2", "moves": _SNORLAX_MOVES, "ability": "Immunity", "item": "Choice Band", "level": _LEVEL}]}
        )
        damage = _exceeding_damage()
        lines = _leads() + _strike_lines(damage, turn=1)
        inference = _infer(lines, source=source)
        strike = inference.strikes[0]
        self.assertIn("cb-pinned-by-elimination", strike.disqualifiers)
        self.assertFalse(strike.cb_eligible)

    def test_off_model_exceedance_is_rejected(self) -> None:
        cb_max = max(_bodyslam_rolls(cb=True))
        damage = int(cb_max + 0.01 * 330 + 1.0 + 3)
        lines = _leads()
        lines += _strike_lines(damage, turn=1)
        lines += _strike_lines(damage, turn=2, prior_hp=330 - damage)
        inference = _infer(lines)
        self.assertNotIn("p2:snorlax", inference.cb_bits)
        strike = inference.strikes[0]
        self.assertIn("exceeds-cb-explanation", strike.disqualifiers)
        self.assertFalse(strike.cb_exceeded)

    def test_whitelist_gates_cb_counting(self) -> None:
        damage = _exceeding_damage()
        lines = _leads()
        lines += _strike_lines(damage, turn=1)
        lines += _strike_lines(damage, turn=2, prior_hp=330 - damage)
        inference = _infer(lines, whitelist={})
        self.assertNotIn("p2:snorlax", inference.cb_bits)
        self.assertIn("not-whitelisted", inference.strikes[0].disqualifiers)

    def test_ko_clipped_strike_can_still_prove_exceedance(self) -> None:
        # Defender at low HP: the KO'd remainder already exceeds the non-CB max.
        damage = _exceeding_damage()
        prior = damage  # exactly enough that remaining HP == exceeding damage
        lines = _leads(p1_hp=f"{prior}/330")
        lines += [
            "|move|p2a: Snorlax|Body Slam|p1a: Slowbro",
            "|-damage|p1a: Slowbro|0 fnt",
            "|faint|p1a: Slowbro",
            "|",
            "|upkeep",
            "|turn|2",
        ]
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertTrue(strike.cb_eligible)
        self.assertTrue(strike.cb_exceeded)
        self.assertFalse(strike.residual_valid)  # clipped observation, no residual
        self.assertIn("ko-clipped", strike.disqualifiers)

    def test_ko_only_exceedances_never_flip_the_bit(self) -> None:
        # Two KO-clipped exceedances: both count as strikes, but the bit requires at
        # least one NON-KO exceedance (the off-model upper guard is weakened on
        # clipped observations).
        damage = _exceeding_damage()
        ko_block = [
            "|move|p2a: Snorlax|Body Slam|p1a: Slowbro",
            "|-damage|p1a: Slowbro|0 fnt",
            "|faint|p1a: Slowbro",
            "|",
        ]
        lines = _leads(p1_hp=f"{damage}/330")
        lines += ko_block
        lines += [
            f"|switch|p1a: Slowbro|Slowbro, L80|{damage}/330",
            "|upkeep",
            "|turn|2",
        ]
        lines += ko_block + ["|upkeep", "|turn|3"]
        inference = _infer(lines)
        self.assertEqual(len(inference.cb_strike_turns.get("p2:snorlax", ())), 2)
        self.assertFalse(inference.cb_bits.get("p2:snorlax", False))
        # One KO strike + one clean exceedance flips it.
        lines = _leads(p1_hp=f"{damage}/330")
        lines += ko_block
        lines += [
            "|switch|p1a: Slowbro|Slowbro, L80|330/330",
            "|upkeep",
            "|turn|2",
        ]
        lines += _strike_lines(damage, turn=2)
        inference = _infer(lines)
        self.assertTrue(inference.cb_bits.get("p2:snorlax", False))


class ResidualChannelTest(unittest.TestCase):
    def test_residual_sign_and_token_population(self) -> None:
        rolls = _bodyslam_rolls()
        expected = median_damage(rolls)
        high = max(rolls)
        lines = _leads() + _strike_lines(high, turn=1)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.residual_valid)
        self.assertGreater(strike.residual, 0)
        self.assertAlmostEqual(strike.residual, (high - expected) / 330.0)
        token = inference.tokens[strike.token_index]
        self.assertTrue(token.residual_valid)
        self.assertAlmostEqual(token.residual, strike.residual)
        # Our own attacks never carry residuals in base Tier 2.
        own_attack_tokens = [
            t for t in inference.tokens if t.actor_slot == "p1" and t.kind == "move"
        ]
        for token in own_attack_tokens:
            self.assertFalse(token.residual_valid)

    def test_burn_conditioning_and_guts_ambiguity(self) -> None:
        # Burned attacker, no Guts candidates: expected halves deterministically.
        damage = int(median_damage(_bodyslam_rolls(burned=True)))
        lines = _leads()
        lines += ["|-status|p2a: Snorlax|brn"]
        lines += _strike_lines(damage, turn=1)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertTrue(strike.residual_valid)
        self.assertAlmostEqual(
            strike.expected_median_hp, median_damage(_bodyslam_rolls(burned=True))
        )
        # Mixed Guts / non-Guts candidates while burned: baselines disagree -> masked.
        mixed = FakeSource(
            {
                "snorlax": [
                    {"variant_id": "a", "moves": _SNORLAX_MOVES, "ability": "Immunity", "item": "Leftovers", "level": _LEVEL},
                    {"variant_id": "b", "moves": _SNORLAX_MOVES, "ability": "Guts", "item": "Leftovers", "level": _LEVEL},
                ]
            }
        )
        inference = _infer(lines, source=mixed)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertFalse(strike.residual_valid)
        self.assertIn("baseline-disagreement", strike.disqualifiers)

    def test_bonemerang_normalizes_per_hit(self) -> None:
        stats = _snorlax_stats()
        per_hit = gen3_damage_rolls(
            Gen3DamageContext(
                level=_LEVEL, base_power=60, category="Physical",
                attack=stats["atk"], defense=_OWN_SLOWBRO.stats["def"],
                stab=False, effectiveness=1.0,
            )
        )
        source = FakeSource(
            {"snorlax": [{"variant_id": "a", "moves": ["bonemerang", "rest", "sleeptalk", "bodyslam"], "ability": "Immunity", "item": "Leftovers", "level": _LEVEL}]}
        )
        total = 2 * int(median_damage(per_hit))
        first = per_hit[7]
        second = total - first
        lines = _leads()
        lines += [
            "|move|p2a: Snorlax|Bonemerang|p1a: Slowbro",
            f"|-damage|p1a: Slowbro|{330 - first}/330",
            f"|-damage|p1a: Slowbro|{330 - first - second}/330",
            "|-hitcount|p1a: Slowbro|2",
            "|",
            "|upkeep",
            "|turn|2",
        ]
        inference = _infer(lines, source=source)
        strike = next(s for s in inference.strikes if s.move_id == "bonemerang")
        # The per-hit expectation is still computed and exposed, but multi-hit
        # residuals ship INVALID: the summed-roll population is outside the gate's
        # calibrated population (production validity == calibration coverage).
        self.assertAlmostEqual(strike.expected_median_hp, 2 * median_damage(per_hit))
        self.assertFalse(strike.residual_valid)
        self.assertIn("multi-hit", strike.disqualifiers)
        self.assertFalse(strike.cb_eligible)

    def test_plus_minus_cross_field_activation(self) -> None:
        # Gen3 Plus/Minus check ALL actives (mods/gen3/abilities.ts): in singles the
        # partner is OUR active, whose ability is exactly known. Minus attacker into
        # our Plus defender gets 1.5x SpA; into anyone else it stays inert.
        source = FakeSource(
            {"snorlax": [{"variant_id": "m", "moves": ["flamethrower", "rest", "sleeptalk", "bodyslam"], "ability": "Minus", "item": "Leftovers", "level": _LEVEL}]}
        )
        stats = randbats_spread_stats(
            _DEX.species_info("snorlax").base_stats, level=_LEVEL,
            moves=["flamethrower", "rest", "sleeptalk", "bodyslam"], item="Leftovers",
            has_physical_attack=True,
        )

        def rolls(attack_mods=()):
            return gen3_damage_rolls(
                Gen3DamageContext(
                    level=_LEVEL, base_power=95, category="Special",
                    attack=stats["spa"], defense=_OWN_SLOWBRO.stats["spd"],
                    attack_mods=tuple(attack_mods),
                    effectiveness=_DEX.effectiveness("Fire", ("Water", "Psychic")),
                )
            )

        lines = _leads() + _strike_lines(20, turn=1, move="Flamethrower")
        plain = _infer(lines, source=source)
        strike = next(s for s in plain.strikes if s.move_id == "flamethrower")
        self.assertAlmostEqual(strike.expected_median_hp, median_damage(rolls()))
        plus_defender = (
            OwnMon(species="Slowbro", level=80, stats=_OWN_SLOWBRO.stats, ability="Plus", item=None),
        )
        boosted = _infer(lines, source=source, own_team=plus_defender)
        strike = next(s for s in boosted.strikes if s.move_id == "flamethrower")
        self.assertAlmostEqual(strike.expected_median_hp, median_damage(rolls([(1.5, 1)])))

    def test_transformed_attacker_is_excluded(self) -> None:
        lines = _leads()
        lines += ["|-transform|p2a: Snorlax|p1a: Slowbro"]
        lines += _strike_lines(50, turn=1)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertFalse(strike.residual_valid)
        self.assertIn("transformed-attacker", strike.disqualifiers)

    def test_white_herb_clears_negative_stages(self) -> None:
        # Superpower-style self-drop then White Herb's silent restore: the strike
        # afterwards must NOT carry stale negative stages.
        damage = int(median_damage(_bodyslam_rolls()))
        lines = _leads()
        lines += [
            "|-unboost|p2a: Snorlax|atk|1",
            "|-enditem|p2a: Snorlax|White Herb",
            "|-clearnegativeboost|p2a: Snorlax|[silent]",
        ]
        lines += _strike_lines(damage, turn=1)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertNotIn("stat-stages", strike.disqualifiers)
        self.assertAlmostEqual(strike.expected_median_hp, median_damage(_bodyslam_rolls()))

    def test_forecast_forme_change_excludes_like_type_change(self) -> None:
        lines = _leads()
        lines += ["|-formechange|p2a: Snorlax|Castform-Rainy|[msg]"]
        lines += _strike_lines(50, turn=1)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertFalse(strike.residual_valid)
        self.assertIn("type-changed", strike.disqualifiers)

    def test_traced_ability_disqualifies_until_switch(self) -> None:
        # Trace acquisition (either side) replaces a live ability: no damage
        # inference while the tracer is on the field (review HIGH-2). The rule is
        # structural — it keys on the acquisition-tagged |-ability| line, not on
        # which species traced.
        damage = int(median_damage(_bodyslam_rolls()))
        lines = _leads()
        lines += ["|-ability|p1a: Slowbro|Immunity|[from] ability: Trace|[of] p2a: Snorlax"]
        lines += _strike_lines(damage, turn=1)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertIn("ability-overridden", strike.disqualifiers)
        self.assertFalse(strike.residual_valid)
        self.assertFalse(strike.cb_eligible)
        # Attacker-side trace disqualifies too.
        lines = _leads()
        lines += ["|-ability|p2a: Snorlax|Oblivious|[from] ability: Trace|[of] p1a: Slowbro"]
        lines += _strike_lines(damage, turn=1)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertIn("ability-overridden", strike.disqualifiers)
        # The override ends when the traced mon leaves the field.
        lines = _leads()
        lines += ["|-ability|p2a: Snorlax|Oblivious|[from] ability: Trace|[of] p1a: Slowbro"]
        lines += [
            "|switch|p2a: Flareon|Flareon, L80|100/100",
            "|",
            "|upkeep",
            "|turn|2",
            "|switch|p2a: Snorlax|Snorlax, L80|100/100",
            "|",
            "|upkeep",
            "|turn|3",
        ]
        lines += _strike_lines(damage, turn=3)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertNotIn("ability-overridden", strike.disqualifiers)
        self.assertTrue(strike.residual_valid)

    def test_trick_item_mutation_excludes_and_persists_across_switches(self) -> None:
        damage = _exceeding_damage()
        prefix = _leads() + [
            "|-activate|p2a: Snorlax|move: Trick|[of] p1a: Slowbro",
            "|-item|p1a: Slowbro|Choice Band|[from] move: Trick",
            "|-item|p2a: Snorlax|Leftovers|[from] move: Trick",
        ]
        # Immediately after the swap, both sides' items are mutated.
        lines = prefix + _strike_lines(damage, turn=1)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertIn("item-mutated", strike.disqualifiers)
        self.assertFalse(strike.residual_valid)
        self.assertFalse(strike.cb_eligible)
        # The mutation follows the mon across a switch cycle (item state persists).
        lines = prefix + [
            "|switch|p2a: Flareon|Flareon, L80|100/100",
            "|",
            "|upkeep",
            "|turn|2",
            "|switch|p2a: Snorlax|Snorlax, L80|100/100",
            "|",
            "|upkeep",
            "|turn|3",
        ] + _strike_lines(damage, turn=3)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertIn("item-mutated", strike.disqualifiers)

    def test_type_changed_attacker_is_excluded_until_switch_renews(self) -> None:
        # Color Change (Kecleon-class): after a typechange volatile the attacker's
        # live typing (and thus STAB) no longer follows its species -> excluded.
        lines = _leads()
        lines += ["|-start|p2a: Snorlax|typechange|Water|[from] ability: Color Change"]
        lines += _strike_lines(50, turn=1)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertFalse(strike.residual_valid)
        self.assertIn("type-changed", strike.disqualifiers)
        self.assertFalse(strike.cb_eligible)

        # Switching out clears the volatile: a fresh entry infers normally again.
        lines = _leads()
        lines += ["|-start|p2a: Snorlax|typechange|Water|[from] ability: Color Change"]
        lines += [
            "|switch|p2a: Flareon|Flareon, L80|100/100",
            "|",
            "|upkeep",
            "|turn|2",
            "|switch|p2a: Snorlax|Snorlax, L80|100/100",
            "|",
            "|upkeep",
            "|turn|3",
        ]
        damage = int(median_damage(_bodyslam_rolls()))
        lines += _strike_lines(damage, turn=3)
        inference = _infer(lines)
        strike = next(s for s in inference.strikes if s.move_id == "bodyslam")
        self.assertNotIn("type-changed", strike.disqualifiers)
        self.assertTrue(strike.residual_valid)


class WeatherAndVolatileTest(unittest.TestCase):
    def _charizard_source(self):
        return FakeSource(
            {"charizard": [{"variant_id": "z", "moves": ["flamethrower", "solarbeam", "rest", "sleeptalk"], "ability": "Blaze", "item": "Leftovers", "level": _LEVEL}]}
        )

    def _charizard_rolls(self, move_id, *, base_power, bp_mods=(), attack_mods=(), weather_mod=None):
        stats = randbats_spread_stats(
            _DEX.species_info("charizard").base_stats,
            level=_LEVEL,
            moves=["flamethrower", "solarbeam", "rest", "sleeptalk"],
            item="Leftovers",
            has_physical_attack=False,
        )
        move_type = {"flamethrower": "Fire", "solarbeam": "Grass"}[move_id]
        return gen3_damage_rolls(
            Gen3DamageContext(
                level=_LEVEL, base_power=base_power, category="Special",
                attack=stats["spa"], defense=_OWN_SLOWBRO.stats["spd"],
                base_power_mods=tuple(bp_mods), attack_mods=tuple(attack_mods),
                stab=move_type == "Fire",
                effectiveness=_DEX.effectiveness(move_type, ("Water", "Psychic")),
                weather_mod=weather_mod,
            )
        )

    def _solar_beam_strike(self, weather_line):
        lines = _leads(p2="Charizard")
        if weather_line:
            lines += [weather_line]
        damage = 60
        lines += [
            "|move|p2a: Charizard|Solar Beam|p1a: Slowbro",
            f"|-damage|p1a: Slowbro|{330 - damage}/330",
            "|",
            "|upkeep",
            "|turn|2",
        ]
        inference = _infer(lines, source=self._charizard_source())
        return next(s for s in inference.strikes if s.move_id == "solarbeam")

    def test_solar_beam_halved_in_rain_sand_hail_but_not_sun(self) -> None:
        halved = median_damage(self._charizard_rolls("solarbeam", base_power=120, bp_mods=[(0.5, 1)]))
        full = median_damage(self._charizard_rolls("solarbeam", base_power=120))
        self.assertLess(halved, full)
        for weather in ("RainDance", "Sandstorm", "Hail"):
            strike = self._solar_beam_strike(f"|-weather|{weather}")
            self.assertAlmostEqual(strike.expected_median_hp, halved, msg=weather)
        strike = self._solar_beam_strike("|-weather|SunnyDay")
        self.assertAlmostEqual(strike.expected_median_hp, full)
        strike = self._solar_beam_strike(None)
        self.assertAlmostEqual(strike.expected_median_hp, full)

    def test_rain_weakens_fire_and_flash_fire_boosts_it(self) -> None:
        # Flash Fire volatile: tracked from |-start|, cleared on switch-out.
        source = FakeSource(
            {"flareon": [{"variant_id": "f", "moves": ["flamethrower", "rest", "sleeptalk", "bodyslam"], "ability": "Flash Fire", "item": "Leftovers", "level": _LEVEL}]}
        )
        stats = randbats_spread_stats(
            _DEX.species_info("flareon").base_stats,
            level=_LEVEL,
            moves=["flamethrower", "rest", "sleeptalk", "bodyslam"],
            item="Leftovers",
            has_physical_attack=True,
        )

        def flareon_rolls(phase1_mods=()):
            # Flash Fire's boost is a ModifyDamagePhase1 damage mod (gen4-mod hook,
            # inherited by gen3) — NOT an attack-stat mod.
            return gen3_damage_rolls(
                Gen3DamageContext(
                    level=_LEVEL, base_power=95, category="Special",
                    attack=stats["spa"], defense=_OWN_SLOWBRO.stats["spd"],
                    phase1_mods=tuple(phase1_mods), stab=True,
                    effectiveness=_DEX.effectiveness("Fire", ("Water", "Psychic")),
                )
            )

        lines = _leads(p2="Flareon")
        lines += ["|-start|p2a: Flareon|ability: Flash Fire"]
        lines += [
            "|move|p2a: Flareon|Flamethrower|p1a: Slowbro",
            "|-damage|p1a: Slowbro|290/330",
            "|",
            "|upkeep",
            "|turn|2",
        ]
        inference = _infer(lines, source=source)
        strike = next(s for s in inference.strikes if s.move_id == "flamethrower")
        self.assertAlmostEqual(
            strike.expected_median_hp, median_damage(flareon_rolls([(1.5, 1)]))
        )

        # After switching out and back in, the volatile is gone.
        lines = _leads(p2="Flareon")
        lines += ["|-start|p2a: Flareon|ability: Flash Fire"]
        lines += [
            "|switch|p2a: Snorlax|Snorlax, L80|100/100",
            "|",
            "|upkeep",
            "|turn|2",
            "|switch|p2a: Flareon|Flareon, L80|100/100",
            "|",
            "|upkeep",
            "|turn|3",
            "|move|p2a: Flareon|Flamethrower|p1a: Slowbro",
            "|-damage|p1a: Slowbro|295/330",
            "|",
            "|upkeep",
            "|turn|4",
        ]
        inference = _infer(lines, source=source)
        strike = next(s for s in inference.strikes if s.move_id == "flamethrower")
        self.assertAlmostEqual(strike.expected_median_hp, median_damage(flareon_rolls()))


class ContextFoldTest(unittest.TestCase):
    def test_baton_pass_inherits_boosts_and_switch_clears_them(self) -> None:
        damage = 40
        base_lines = _leads()
        # Boost, then Baton Pass into Flareon: the boost persists.
        bp_lines = base_lines + [
            "|-boost|p2a: Snorlax|atk|2",
            "|move|p2a: Snorlax|Baton Pass|p2a: Snorlax",
            "|switch|p2a: Flareon|Flareon, L80|100/100",
            "|",
            "|upkeep",
            "|turn|2",
            "|move|p2a: Flareon|Body Slam|p1a: Slowbro",
            f"|-damage|p1a: Slowbro|{330 - damage}/330",
            "|",
            "|upkeep",
            "|turn|3",
        ]
        source = FakeSource(
            {
                "snorlax": _SNORLAX_VARIANTS,
                "flareon": [{"variant_id": "f", "moves": ["bodyslam", "rest", "sleeptalk", "flamethrower"], "ability": "Flash Fire", "item": "Leftovers", "level": _LEVEL}],
            }
        )
        inference = _infer(bp_lines, source=source, whitelist={"flareon": frozenset({"bodyslam"})})
        strike = next(s for s in inference.strikes if s.attacker_key == "p2:flareon")
        self.assertIn("stat-stages", strike.disqualifiers)

        # A plain switch clears the boost: same lines without the Baton Pass move.
        plain_lines = base_lines + [
            "|-boost|p2a: Snorlax|atk|2",
            "|switch|p2a: Flareon|Flareon, L80|100/100",
            "|",
            "|upkeep",
            "|turn|2",
            "|move|p2a: Flareon|Body Slam|p1a: Slowbro",
            f"|-damage|p1a: Slowbro|{330 - damage}/330",
            "|",
            "|upkeep",
            "|turn|3",
        ]
        inference = _infer(plain_lines, source=source, whitelist={"flareon": frozenset({"bodyslam"})})
        strike = next(s for s in inference.strikes if s.attacker_key == "p2:flareon")
        self.assertNotIn("stat-stages", strike.disqualifiers)

    def test_brick_break_strips_screen_from_its_own_strike_only(self) -> None:
        # Fold-level guarantee behind the residual/CB fix: a screen-shattering move's own
        # strike context drops Reflect (its onTryHit removes screens before it hits), while
        # a same-screen non-shatter strike keeps it and the NEXT strike sees the shatter.
        bodyslam = "|move|p2a: Snorlax|Body Slam|p1a: Slowbro"
        brickbreak = "|move|p2a: Snorlax|Brick Break|p1a: Slowbro"
        lines = _leads()
        lines += [
            "|move|p1a: Slowbro|Reflect|p1a: Slowbro",
            "|-sidestart|p1: Alice|Reflect",
            "|", "|upkeep", "|turn|2",
            # Non-shatter physical strike into the live screen: keeps reflect.
            bodyslam,
            "|-damage|p1a: Slowbro|300/330",
            "|", "|upkeep", "|turn|3",
            # Brick Break shatters the screen (onTryHit) before its own damage lands.
            brickbreak,
            "|-sideend|p1: Alice|Reflect",
            "|-damage|p1a: Slowbro|270/330",
            "|", "|upkeep", "|turn|4",
            # Strike after the shatter: reflect is gone from the fold state too.
            bodyslam,
            "|-damage|p1a: Slowbro|240/330",
            "|", "|upkeep", "|turn|5",
        ]
        idx_before = lines.index(bodyslam)
        idx_brick = lines.index(brickbreak)
        idx_after = len(lines) - 1 - lines[::-1].index(bodyslam)
        self.assertLess(idx_before, idx_brick)
        self.assertLess(idx_brick, idx_after)

        fold = _IncrementalContextFold()
        fold.process(lines)
        self.assertEqual(fold.contexts[idx_before].defender_screens, ("reflect",))
        self.assertEqual(fold.contexts[idx_brick].defender_screens, ())
        self.assertEqual(fold.contexts[idx_after].defender_screens, ())

    def test_transform_copies_target_boosts_in_incremental_fold(self) -> None:
        # Keep Tier 2's incremental public ledger aligned with the replay observation state.
        fold = _IncrementalContextFold()
        fold.process(
            [
                "|-boost|p1a: Slowbro|spa|2",
                "|-boost|p1a: Slowbro|spd|1",
                "|-transform|p2a: Ditto|p1a: Slowbro",
            ]
        )
        self.assertTrue(fold.transformed["p2"])
        self.assertEqual(fold.boosts["p2"], {"spa": 2, "spd": 1})


class WhitelistTest(unittest.TestCase):
    def _universe(self, species, variants):
        return SimpleNamespace(
            species=species,
            level=_LEVEL,
            variants=[
                SimpleNamespace(
                    moves=v["moves"], ability=v.get("ability", ""), item=v.get("item", ""), level=v.get("level", _LEVEL)
                )
                for v in variants
            ],
        )

    def test_inclusion_and_class_exclusions(self) -> None:
        universes = {
            "snorlax": self._universe(
                "Snorlax",
                [
                    {"moves": ("bodyslam", "earthquake", "return", "flail")},
                    {"moves": ("bodyslam", "counter", "bonemerang", "hiddenpowergrass")},
                    {"moves": ("bodyslam", "pursuit", "flamethrower", "rockslide")},
                ],
            )
        }
        whitelist = build_cb_whitelist(universes, _DEX)
        eligible = whitelist["snorlax"]
        self.assertIn("bodyslam", eligible)
        self.assertIn("earthquake", eligible)
        self.assertIn("return", eligible)  # fixed 102 despite dex BP 0
        self.assertIn("rockslide", eligible)
        self.assertNotIn("flail", eligible)  # HP-scaled
        self.assertNotIn("counter", eligible)  # fixed-damage callback
        self.assertNotIn("bonemerang", eligible)  # multi-hit
        self.assertNotIn("pursuit", eligible)  # scenario-scaled (and special in gen3)
        self.assertNotIn("hiddenpowergrass", eligible)  # IV-scaled
        self.assertNotIn("flamethrower", eligible)  # special

    def test_unpinned_attack_stat_excludes_the_pair(self) -> None:
        universes = {
            "snorlax": self._universe(
                "Snorlax",
                [
                    {"moves": ("bodyslam", "earthquake"), "level": 80},
                    {"moves": ("bodyslam", "rest"), "level": 90},  # different level -> different Atk
                ],
            )
        }
        whitelist = build_cb_whitelist(universes, _DEX)
        self.assertNotIn("bodyslam", whitelist.get("snorlax", frozenset()))
        self.assertIn("earthquake", whitelist["snorlax"])  # single-carrier: pinned


class ObservationWiringTest(unittest.TestCase):
    """Encode-level wiring: Tier-2 residuals flow into #502's reserved v2 slots."""

    def test_residual_slots_encode_from_tier2_tokens(self) -> None:
        from dataclasses import replace as dataclass_replace

        from pokezero.category_vocab import build_category_vocabulary
        from pokezero.observation import ObservationFeatureMasks
        from pokezero.showdown import (
            NUMERIC_TT_RESIDUAL,
            NUMERIC_TT_RESIDUAL_VALID,
            TRANSITION_TOKEN_OFFSET,
            V2_1_REPLAY_OBSERVATION_SPEC,
            normalize_for_player,
            observation_from_player_state,
        )

        damage = max(_bodyslam_rolls())
        lines = _leads() + _strike_lines(damage, turn=1)
        request_line = (
            '|request|{"active":[{"moves":[{"move":"Surf","id":"surf"}]}],'
            '"side":{"id":"p1","name":"Alice","pokemon":[{"ident":"p1a: Slowbro",'
            '"details":"Slowbro, L80","condition":"330/330","active":true}]}}'
        )
        full = lines[:2] + [request_line] + lines[2:]
        replay = parse_showdown_replay(full)
        inference = _infer(full)
        strike = next(s for s in inference.strikes if s.residual_valid)

        state = normalize_for_player(replay, player_id="agent", player_name="Alice")
        wired = dataclass_replace(
            state, transition_tokens=apply_residuals(state.transition_tokens, inference)
        )
        vocab = build_category_vocabulary(
            [
                "species:Slowbro", "species:Snorlax", "move:bodyslam",
                "transition:self", "transition:opponent",
                "tt_kind:move", "tt_kind:switch",
                "tt_outcome:normal", "tt_effectiveness:neutral", "tt_side_effect:none",
            ]
        )
        row_index = TRANSITION_TOKEN_OFFSET + strike.token_index

        observation = observation_from_player_state(
            wired, category_vocab=vocab, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        row = observation.numeric_features[row_index]
        self.assertAlmostEqual(row[NUMERIC_TT_RESIDUAL], max(-1.0, min(1.0, strike.residual)))
        self.assertEqual(row[NUMERIC_TT_RESIDUAL_VALID], 1.0)

        # The tier2_residuals mask darkens the channel without touching anything else.
        masked = observation_from_player_state(
            wired, category_vocab=vocab, feature_masks=ObservationFeatureMasks(tier2_residuals=False),
            spec=V2_1_REPLAY_OBSERVATION_SPEC,
        )
        self.assertEqual(masked.numeric_features[row_index][NUMERIC_TT_RESIDUAL], 0.0)
        self.assertEqual(masked.numeric_features[row_index][NUMERIC_TT_RESIDUAL_VALID], 0.0)
        self.assertEqual(
            masked.numeric_features[row_index][: NUMERIC_TT_RESIDUAL],
            row[: NUMERIC_TT_RESIDUAL],
        )

        # Plain-extraction tokens carry no residuals: the slots stay zero.
        plain = observation_from_player_state(
            state, category_vocab=vocab, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        self.assertEqual(plain.numeric_features[row_index][NUMERIC_TT_RESIDUAL], 0.0)
        self.assertEqual(plain.numeric_features[row_index][NUMERIC_TT_RESIDUAL_VALID], 0.0)

    def test_cb_pinned_encodes_on_the_opp_mon_row_from_a_real_conclusion(self) -> None:
        """v2.1 per-mon pinned surface, wired from a REAL two-strike inference (not a
        hand-stamped token): pinned on Snorlax's opp-mon row, dark under the mask, and
        the investment twin stays a reserve. Layer separation: the conclusion reaches
        the observation through the tier2 token annotation only — the Tier-1 belief
        engine's candidate sets are never written by tier2 (no belief-column change)."""
        from dataclasses import replace as dataclass_replace

        from pokezero.category_vocab import build_category_vocabulary
        from pokezero.observation import ObservationFeatureMasks
        from pokezero.showdown import (
            NUMERIC_TIER2_CB_PINNED,
            NUMERIC_TIER2_INVESTMENT_PINNED,
            OPPONENT_POKEMON_TOKEN_OFFSET,
            V2_1_REPLAY_OBSERVATION_SPEC,
            normalize_for_player,
            observation_from_player_state,
        )

        damage = _exceeding_damage()
        lines = _leads()
        lines += _strike_lines(damage, turn=1)
        lines += _strike_lines(damage, turn=2, prior_hp=330 - damage)
        request_line = (
            '|request|{"active":[{"moves":[{"move":"Surf","id":"surf"}]}],'
            '"side":{"id":"p1","name":"Alice","pokemon":[{"ident":"p1a: Slowbro",'
            '"details":"Slowbro, L80","condition":"330/330","active":true}]}}'
        )
        full = lines[:2] + [request_line] + lines[2:]
        inference = _infer(full)
        self.assertTrue(inference.cb_bits.get("p2:snorlax"))

        replay = parse_showdown_replay(full)
        state = normalize_for_player(replay, player_id="agent", player_name="Alice")
        wired = dataclass_replace(
            state, transition_tokens=apply_residuals(state.transition_tokens, inference)
        )
        vocab = build_category_vocabulary(
            [
                "species:Slowbro", "species:Snorlax", "move:bodyslam",
                "transition:self", "transition:opponent",
                "tt_kind:move", "tt_kind:switch",
                "tt_outcome:normal", "tt_effectiveness:neutral", "tt_side_effect:none",
            ]
        )
        snorlax_index = next(
            index for index, mon in enumerate(state.opponent_team) if mon.species == "Snorlax"
        )
        observation = observation_from_player_state(
            wired, category_vocab=vocab, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET + snorlax_index]
        self.assertEqual(row[NUMERIC_TIER2_CB_PINNED], 1.0)
        self.assertEqual(row[NUMERIC_TIER2_INVESTMENT_PINNED], 0.0)

        masked = observation_from_player_state(
            wired, category_vocab=vocab, feature_masks=ObservationFeatureMasks(tier2_residuals=False),
            spec=V2_1_REPLAY_OBSERVATION_SPEC,
        )
        self.assertEqual(
            masked.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET + snorlax_index][
                NUMERIC_TIER2_CB_PINNED
            ],
            0.0,
        )


class HelpersTest(unittest.TestCase):
    def test_own_team_from_request(self) -> None:
        request = {
            "side": {
                "id": "p1",
                "pokemon": [
                    {
                        "details": "Slowbro, L80, M",
                        "condition": "330/330",
                        "stats": {"atk": 150, "def": 230, "spa": 210, "spd": 180, "spe": 100},
                        "baseAbility": "oblivious",
                        "item": "leftovers",
                        "moves": ["surf"],
                    }
                ],
            }
        }
        team = own_team_from_request(request)
        self.assertEqual(len(team), 1)
        mon = team[0]
        self.assertEqual(mon.species, "Slowbro")
        self.assertEqual(mon.level, 80)
        self.assertEqual(mon.stats["hp"], 330)
        self.assertEqual(mon.stats["def"], 230)
        self.assertEqual(mon.ability, "oblivious")
        self.assertEqual(mon.item, "leftovers")

    def test_variant_has_physical_attack(self) -> None:
        self.assertTrue(variant_has_physical_attack(("bodyslam", "rest"), _DEX))
        self.assertTrue(variant_has_physical_attack(("return",), _DEX))  # BP-callback physical
        self.assertFalse(variant_has_physical_attack(("counter", "rest"), _DEX))  # fixed damage
        self.assertFalse(variant_has_physical_attack(("flamethrower",), _DEX))

    def test_apply_residuals_copies_fields(self) -> None:
        damage = int(median_damage(_bodyslam_rolls()))
        lines = _leads() + _strike_lines(damage, turn=1)
        replay = parse_showdown_replay(lines)
        inference = _infer(lines)
        tokens = extract_transition_tokens(replay, perspective_slot="p1")
        updated = apply_residuals(tokens, inference)
        self.assertEqual(
            [(t.residual, t.residual_valid, t.cb_bit) for t in updated],
            [(t.residual, t.residual_valid, t.cb_bit) for t in inference.tokens],
        )
        with self.assertRaises(ValueError):
            apply_residuals(tokens[:-1], inference)




class LiveTrackerTest(unittest.TestCase):
    """The incremental live consumer must agree with the batch inference at the
    observation boundaries the env actually annotates at."""

    def _multi_turn_lines(self):
        damage = _exceeding_damage()
        median = int(median_damage(_bodyslam_rolls()))
        lines = _leads()
        # Turn 1: clean exceedance; turn 2: crit strike (residual crit-conditioned);
        # turn 3: boosted strike (stat-stages CB disqualifier, residual conditioned);
        # turn 4: second clean exceedance -> CB bit flips.
        lines += _strike_lines(damage, turn=1)
        prior = 330 - damage
        crit = int(median_damage(_bodyslam_rolls(crit=True)))
        lines += [
            "|move|p2a: Snorlax|Body Slam|p1a: Slowbro",
            "|-crit|p1a: Slowbro",
            f"|-damage|p1a: Slowbro|{max(1, prior - crit)}/330",
            "|",
            "|upkeep",
            "|turn|3",
            "|-boost|p2a: Snorlax|atk|1",
            "|move|p2a: Snorlax|Body Slam|p1a: Slowbro",
            f"|-damage|p1a: Slowbro|{max(1, prior - crit - median)}/330",
            "|",
            "|upkeep",
            "|turn|4",
            "|-unboost|p2a: Snorlax|atk|1",
        ]
        # Heal to full so the second exceedance is unclipped, then strike again.
        lines += [
            "|move|p1a: Slowbro|Recover|p1a: Slowbro",
            "|-heal|p1a: Slowbro|330/330",
            "|move|p2a: Snorlax|Body Slam|p1a: Slowbro",
            f"|-damage|p1a: Slowbro|{330 - damage}/330",
            "|",
            "|upkeep",
            "|turn|5",
        ]
        return lines

    def _turn_boundaries(self, lines):
        """Line-count prefixes at each |turn| boundary plus the full log (the env's
        observation points)."""
        boundaries = []
        for index, line in enumerate(lines):
            if line.startswith("|turn|"):
                boundaries.append(index + 1)
        if not boundaries or boundaries[-1] != len(lines):
            boundaries.append(len(lines))
        return boundaries

    def test_incremental_annotation_matches_batch_inference(self) -> None:
        from pokezero.belief import PublicBattleBeliefEngine

        lines = self._multi_turn_lines()
        source = FakeSource({"snorlax": _SNORLAX_VARIANTS})
        tracker = Tier2LiveTracker(
            perspective_slot="p1",
            own_team=_OWN_TEAM,
            dex=_DEX,
            whitelist=_WHITELIST,
        )
        engine = PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=source)
        fed = 0
        annotated = ()
        for boundary in self._turn_boundaries(lines):
            replay = parse_showdown_replay(lines[:boundary])
            events = replay.public_events
            while fed < len(events):
                engine.ingest_event(events[fed])
                fed += 1
            tokens = extract_transition_tokens(replay, perspective_slot="p1")
            annotated = tracker.annotate(replay, tokens, engine)

        full_replay = parse_showdown_replay(lines)
        batch = infer_tier2(
            full_replay,
            perspective_slot="p1",
            own_team=_OWN_TEAM,
            dex=_DEX,
            set_source=FakeSource({"snorlax": _SNORLAX_VARIANTS}),
            whitelist=_WHITELIST,
        )
        self.assertEqual(len(annotated), len(batch.tokens))
        live_fields = [(t.residual, t.residual_valid, t.cb_bit) for t in annotated]
        batch_fields = [(t.residual, t.residual_valid, t.cb_bit) for t in batch.tokens]
        self.assertEqual(live_fields, batch_fields)
        # The as-of-strike bit is monotone: no True before the concluding strike.
        cb_flags = [t.cb_bit for t in annotated if t.kind == "move" and t.actor_slot == "p2"]
        self.assertIn(True, cb_flags)
        first_true = cb_flags.index(True)
        self.assertTrue(all(cb_flags[first_true:]))
        # The corpus exercises all three strike classes.
        self.assertGreaterEqual(sum(1 for _, valid, _bit in live_fields if valid), 3)
        self.assertEqual(tracker.cb_bits, dict(batch.cb_bits))
        self.assertTrue(tracker.cb_bits.get("p2:snorlax"))

    def test_clone_isolates_nonempty_annotation_state(self) -> None:
        from pokezero.belief import PublicBattleBeliefEngine

        lines = self._multi_turn_lines()
        tracker = Tier2LiveTracker(
            perspective_slot="p1",
            own_team=_OWN_TEAM,
            dex=_DEX,
            whitelist=_WHITELIST,
        )
        engine = PublicBattleBeliefEngine(
            format_id="gen3randombattle",
            set_source=FakeSource({"snorlax": _SNORLAX_VARIANTS}),
        )
        fed = 0
        for boundary in self._turn_boundaries(lines):
            replay = parse_showdown_replay(lines[:boundary])
            while fed < len(replay.public_events):
                engine.ingest_event(replay.public_events[fed])
                fed += 1
            tracker.annotate(
                replay,
                extract_transition_tokens(replay, perspective_slot="p1"),
                engine,
            )

        self.assertTrue(tracker._residuals)
        self.assertTrue(tracker._cb_turns)
        self.assertTrue(tracker._cb_bit_indices)
        original_residuals = dict(tracker._residuals)
        original_cb_turns = {key: list(turns) for key, turns in tracker._cb_turns.items()}
        original_non_ko = set(tracker._cb_non_ko)
        original_bit_indices = set(tracker._cb_bit_indices)
        original_boosts = {side: dict(boosts) for side, boosts in tracker._fold.boosts.items()}

        cloned = tracker.clone()
        attacker_key = next(iter(cloned._cb_turns))
        cloned._residuals[-1] = 1.0
        cloned._cb_turns[attacker_key].append(99)
        cloned._cb_non_ko.add("p2:clone-only")
        cloned._cb_bit_indices.add(999)
        cloned._fold.boosts["p1"]["atk"] = 6

        self.assertEqual(tracker._residuals, original_residuals)
        self.assertEqual(tracker._cb_turns, original_cb_turns)
        self.assertEqual(tracker._cb_non_ko, original_non_ko)
        self.assertEqual(tracker._cb_bit_indices, original_bit_indices)
        self.assertEqual(tracker._fold.boosts, original_boosts)

    def test_annotate_rejects_misaligned_tokens(self) -> None:
        from pokezero.belief import PublicBattleBeliefEngine

        lines = _leads() + _strike_lines(40, turn=1)
        replay = parse_showdown_replay(lines)
        tokens = extract_transition_tokens(replay, perspective_slot="p1")
        tracker = Tier2LiveTracker(
            perspective_slot="p1", own_team=_OWN_TEAM, dex=_DEX, whitelist=_WHITELIST
        )
        engine = PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=None)
        for event in replay.public_events:
            engine.ingest_event(event)
        with self.assertRaisesRegex(ValueError, "do not align"):
            tracker.annotate(replay, tokens[:-1], engine)


if __name__ == "__main__":
    unittest.main()
