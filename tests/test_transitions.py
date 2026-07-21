from pathlib import Path
import unittest

from pokezero.showdown import parse_showdown_replay
from pokezero.transitions import (
    DAMAGE_OUTCOME_ABSORBED,
    DAMAGE_OUTCOME_BLOCKED,
    DAMAGE_OUTCOME_BROKE_SUB,
    DAMAGE_OUTCOME_ENDURED,
    DAMAGE_OUTCOME_HIT_SUB,
    DAMAGE_OUTCOME_IMMUNE,
    DAMAGE_OUTCOME_NORMAL,
    EFFECTIVENESS_IMMUNE,
    EFFECTIVENESS_SUPER,
    SIDE_EFFECT_BOOST,
    SIDE_EFFECT_CHARGING,
    SIDE_EFFECT_DRAIN,
    SIDE_EFFECT_HAZARD_CLEAR,
    SIDE_EFFECT_HAZARD_SET,
    SIDE_EFFECT_HEAL,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_STATUS_INFLICTED,
    SIDE_EFFECT_WEATHER_SET,
    TOKEN_KIND_CANT,
    TOKEN_KIND_MOVE,
    TOKEN_KIND_SWITCH,
    extract_tendency_stats,
    extract_transition_tokens,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "showdown"


def fixture_lines(name: str) -> list[str]:
    return (FIXTURE_ROOT / name).read_text().splitlines()


def _leads(p1_species: str = "Tyranitar", p2_species: str = "Alakazam") -> list[str]:
    return [
        "|player|p1|Alice|",
        "|player|p2|Bob|",
        f"|switch|p1a: {p1_species}|{p1_species}, L74|100/100",
        f"|switch|p2a: {p2_species}|{p2_species}, L72|100/100",
        "|turn|1",
    ]


def _tokens(lines: list[str], perspective_slot: str = "p1"):
    replay = parse_showdown_replay(lines)
    return extract_transition_tokens(replay, perspective_slot=perspective_slot)


def _moves_only(tokens):
    return [token for token in tokens if token.kind == TOKEN_KIND_MOVE]


class PlainAttackTest(unittest.TestCase):
    def test_plain_attack_token_fields(self) -> None:
        tokens = _tokens(
            _leads()
            + [
                "|move|p1a: Tyranitar|Rock Slide|p2a: Alakazam",
                "|-damage|p2a: Alakazam|55/100",
                "|upkeep",
                "|turn|2",
            ]
        )
        self.assertEqual(len(tokens), 3)  # two lead switch tokens + one move token
        move = tokens[2]
        self.assertEqual(move.kind, TOKEN_KIND_MOVE)
        self.assertEqual(move.turn, 1)
        self.assertEqual(move.actor_slot, "p1")
        self.assertEqual(move.actor_species, "Tyranitar")
        self.assertEqual(move.action, "rockslide")
        self.assertAlmostEqual(move.damage_fraction, 0.45)
        self.assertEqual(move.damage_outcome, DAMAGE_OUTCOME_NORMAL)
        self.assertEqual(move.n_hits, 1)
        self.assertFalse(move.crit)
        self.assertFalse(move.miss)
        self.assertFalse(move.ko)
        self.assertFalse(move.called)
        self.assertFalse(move.transformed)
        self.assertEqual(move.side_effect, SIDE_EFFECT_NONE)
        # Tier-2 reserved fields stay unpopulated in Tier 1.
        self.assertIsNone(move.residual)
        self.assertFalse(move.residual_valid)

    def test_crit_and_effectiveness_flags(self) -> None:
        tokens = _tokens(
            _leads()
            + [
                "|move|p1a: Tyranitar|Rock Slide|p2a: Alakazam",
                "|-supereffective|p2a: Alakazam",
                "|-crit|p2a: Alakazam",
                "|-damage|p2a: Alakazam|10/100",
                "|upkeep",
                "|turn|2",
            ]
        )
        move = tokens[2]
        self.assertTrue(move.crit)
        self.assertEqual(move.effectiveness, EFFECTIVENESS_SUPER)
        self.assertAlmostEqual(move.damage_fraction, 0.90)

    def test_miss_leaves_no_damage(self) -> None:
        tokens = _tokens(
            _leads()
            + [
                "|move|p1a: Tyranitar|Rock Slide|p2a: Alakazam|[miss]",
                "|-miss|p1a: Tyranitar|p2a: Alakazam",
                "|upkeep",
                "|turn|2",
            ]
        )
        move = tokens[2]
        self.assertTrue(move.miss)
        self.assertEqual(move.damage_fraction, 0.0)
        self.assertEqual(move.damage_outcome, DAMAGE_OUTCOME_NORMAL)

    def test_ko_flag_on_move_damage_faint(self) -> None:
        tokens = _tokens(
            _leads()
            + [
                "|move|p1a: Tyranitar|Rock Slide|p2a: Alakazam",
                "|-damage|p2a: Alakazam|0 fnt",
                "|faint|p2a: Alakazam",
                "|upkeep",
                "|switch|p2a: Starmie|Starmie, L76|100/100",
                "|turn|2",
            ]
        )
        move = tokens[2]
        self.assertTrue(move.ko)
        self.assertAlmostEqual(move.damage_fraction, 1.0)
        # The faint-replacement emits its own switch token.
        replacement = tokens[3]
        self.assertEqual(replacement.kind, TOKEN_KIND_SWITCH)
        self.assertEqual(replacement.action, "Starmie")


class MultiHitTest(unittest.TestCase):
    def test_bonemerang_hitcount_and_summed_fraction(self) -> None:
        tokens = _tokens(
            _leads("Marowak", "Blissey")
            + [
                "|move|p1a: Marowak|Bonemerang|p2a: Blissey",
                "|-damage|p2a: Blissey|80/100",
                "|-damage|p2a: Blissey|60/100",
                "|-hitcount|p2a: Blissey|2",
                "|upkeep",
                "|turn|2",
            ]
        )
        move = tokens[2]
        self.assertEqual(move.n_hits, 2)
        self.assertAlmostEqual(move.damage_fraction, 0.40)


class DamageOutcomeTest(unittest.TestCase):
    def test_protect_block(self) -> None:
        tokens = _tokens(
            _leads("Machamp", "Blissey")
            + [
                "|move|p2a: Blissey|Protect|p2a: Blissey",
                "|-singleturn|p2a: Blissey|Protect",
                "|move|p1a: Machamp|Cross Chop|p2a: Blissey",
                "|-activate|p2a: Blissey|move: Protect",
                "|upkeep",
                "|turn|2",
            ]
        )
        protect, cross_chop = _moves_only(tokens)
        self.assertEqual(protect.damage_outcome, DAMAGE_OUTCOME_NORMAL)
        self.assertEqual(cross_chop.damage_outcome, DAMAGE_OUTCOME_BLOCKED)
        self.assertEqual(cross_chop.damage_fraction, 0.0)

    def test_substitute_hit_and_break(self) -> None:
        tokens = _tokens(
            _leads("Swampert", "Zapdos")
            + [
                "|move|p2a: Zapdos|Substitute|p2a: Zapdos",
                "|-start|p2a: Zapdos|Substitute",
                "|-damage|p2a: Zapdos|75/100",
                "|move|p1a: Swampert|Surf|p2a: Zapdos",
                "|-activate|p2a: Zapdos|Substitute|[damage]",
                "|upkeep",
                "|turn|2",
                "|move|p1a: Swampert|Surf|p2a: Zapdos",
                "|-end|p2a: Zapdos|Substitute",
                "|upkeep",
                "|turn|3",
            ]
        )
        substitute, surf_hit, surf_break = _moves_only(tokens)
        self.assertEqual(surf_hit.damage_outcome, DAMAGE_OUTCOME_HIT_SUB)
        self.assertEqual(surf_hit.damage_fraction, 0.0)
        self.assertEqual(surf_break.damage_outcome, DAMAGE_OUTCOME_BROKE_SUB)
        # The self-targeted Substitute cost is untagged damage on the (self) defender.
        self.assertAlmostEqual(substitute.damage_fraction, 0.25)

    def test_immune(self) -> None:
        tokens = _tokens(
            _leads("Golem", "Gengar")
            + [
                "|move|p1a: Golem|Earthquake|p2a: Gengar",
                "|-immune|p2a: Gengar",
                "|upkeep",
                "|turn|2",
            ]
        )
        move = tokens[2]
        self.assertEqual(move.damage_outcome, DAMAGE_OUTCOME_IMMUNE)
        self.assertEqual(move.effectiveness, EFFECTIVENESS_IMMUNE)

    def test_absorbed_via_immune_and_heal_forms(self) -> None:
        immune_form = _tokens(
            _leads("Zapdos", "Lanturn")
            + [
                "|move|p1a: Zapdos|Thunderbolt|p2a: Lanturn",
                "|-immune|p2a: Lanturn|[from] ability: Volt Absorb",
                "|upkeep",
                "|turn|2",
            ]
        )
        self.assertEqual(immune_form[2].damage_outcome, DAMAGE_OUTCOME_ABSORBED)
        heal_form = _tokens(
            _leads("Zapdos", "Lanturn")
            + [
                "|move|p1a: Zapdos|Thunderbolt|p2a: Lanturn",
                "|-heal|p2a: Lanturn|100/100|[from] ability: Volt Absorb|[of] p1a: Zapdos",
                "|upkeep",
                "|turn|2",
            ]
        )
        self.assertEqual(heal_form[2].damage_outcome, DAMAGE_OUTCOME_ABSORBED)

    def test_endured(self) -> None:
        tokens = _tokens(
            _leads("Tyranitar", "Heracross")
            + [
                "|move|p2a: Heracross|Endure|p2a: Heracross",
                "|-singleturn|p2a: Heracross|move: Endure",
                "|move|p1a: Tyranitar|Rock Slide|p2a: Heracross",
                "|-activate|p2a: Heracross|move: Endure",
                "|-damage|p2a: Heracross|1/100",
                "|upkeep",
                "|turn|2",
            ]
        )
        rock_slide = _moves_only(tokens)[1]
        self.assertEqual(rock_slide.damage_outcome, DAMAGE_OUTCOME_ENDURED)
        self.assertAlmostEqual(rock_slide.damage_fraction, 0.99)


class ChipDamageAttributionTest(unittest.TestCase):
    def test_chip_damage_never_reaches_token_fractions(self) -> None:
        tokens = _tokens(
            _leads("Skarmory", "Milotic")
            + [
                "|move|p1a: Skarmory|Toxic|p2a: Milotic",
                "|-status|p2a: Milotic|tox",
                "|-damage|p2a: Milotic|94/100 tox|[from] psn",
                "|upkeep",
                "|turn|2",
            ]
        )
        toxic = tokens[2]
        self.assertEqual(toxic.damage_fraction, 0.0)
        self.assertEqual(toxic.side_effect, SIDE_EFFECT_STATUS_INFLICTED)

    def test_chip_faint_is_not_a_move_ko(self) -> None:
        tokens = _tokens(
            _leads("Skarmory", "Milotic")
            + [
                "|move|p1a: Skarmory|Drill Peck|p2a: Milotic",
                "|-damage|p2a: Milotic|4/100 tox",
                "|-damage|p2a: Milotic|0 fnt|[from] psn",
                "|faint|p2a: Milotic",
                "|upkeep",
                "|switch|p2a: Starmie|Starmie, L76|100/100",
                "|turn|2",
            ]
        )
        drill_peck = tokens[2]
        self.assertFalse(drill_peck.ko)
        self.assertAlmostEqual(drill_peck.damage_fraction, 0.96)


class SwitchTokenTest(unittest.TestCase):
    def test_lead_send_outs_and_voluntary_switch_emit_tokens(self) -> None:
        tokens = _tokens(
            _leads()
            + [
                "|switch|p2a: Starmie|Starmie, L76|100/100",
                "|move|p1a: Tyranitar|Rock Slide|p2a: Starmie",
                "|-damage|p2a: Starmie|70/100",
                "|upkeep",
                "|turn|2",
            ]
        )
        self.assertEqual(
            [token.kind for token in tokens],
            [TOKEN_KIND_SWITCH, TOKEN_KIND_SWITCH, TOKEN_KIND_SWITCH, TOKEN_KIND_MOVE],
        )
        lead = tokens[0]
        self.assertEqual(lead.turn, 0)
        self.assertEqual(lead.actor_slot, "p1")
        self.assertEqual(lead.actor_species, "Tyranitar")
        self.assertEqual(lead.action, "Tyranitar")
        voluntary = tokens[2]
        self.assertEqual(voluntary.turn, 1)
        self.assertEqual(voluntary.action, "Starmie")

    def test_drag_emits_no_token(self) -> None:
        tokens = _tokens(
            _leads("Skarmory", "Milotic")
            + [
                "|move|p1a: Skarmory|Roar|p2a: Milotic",
                "|drag|p2a: Blissey|Blissey, L68|100/100",
                "|upkeep",
                "|turn|2",
            ]
        )
        self.assertEqual([token.kind for token in tokens[2:]], [TOKEN_KIND_MOVE])

    def test_baton_pass_completion_emits_switch_token(self) -> None:
        tokens = _tokens(
            _leads("Celebi", "Milotic")
            + [
                "|move|p1a: Celebi|Baton Pass|p1a: Celebi",
                "|switch|p1a: Zapdos|Zapdos, L75|100/100|[from] Baton Pass",
                "|upkeep",
                "|turn|2",
            ]
        )
        baton_pass, completion = tokens[2], tokens[3]
        self.assertEqual(baton_pass.kind, TOKEN_KIND_MOVE)
        self.assertEqual(completion.kind, TOKEN_KIND_SWITCH)
        self.assertEqual(completion.action, "Zapdos")

    def test_nicknamed_baton_passer_switch_stays_voluntary(self) -> None:
        # Regression (review F6): Baton Pass detection must check the protocol tag
        # fields, not the whole line — a NICKNAME containing "Baton Passer" is not a
        # Baton Pass completion and must still count as a voluntary switch.
        lines = _leads("Skarmory", "Milotic") + [
            "|switch|p2a: Baton Passer|Starmie, L76|100/100",
            "|upkeep",
            "|turn|2",
        ]
        replay = parse_showdown_replay(lines)
        tokens = extract_transition_tokens(replay, perspective_slot="p1")
        self.assertEqual(tokens[2].kind, TOKEN_KIND_SWITCH)
        self.assertEqual(tokens[2].action, "Starmie")
        stats = extract_tendency_stats(replay, perspective_slot="p1")
        self.assertEqual(stats.opponent_switch_count, 1)


class SleepTalkTest(unittest.TestCase):
    def test_rest_talk_turn_emits_three_tokens_with_called_bit(self) -> None:
        # The engine's real Sleep Talk shape is THREE lines: |cant|slp + the click +
        # the called execution (verified against the captured audit games).
        for from_tag in ("[from] Sleep Talk", "[from]move: Sleep Talk"):
            with self.subTest(from_tag=from_tag):
                tokens = _tokens(
                    _leads("Snorlax", "Skarmory")
                    + [
                        "|cant|p1a: Snorlax|slp",
                        "|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax",
                        f"|move|p1a: Snorlax|Body Slam|p2a: Skarmory|{from_tag}",
                        "|-damage|p2a: Skarmory|70/100",
                        "|upkeep",
                        "|turn|2",
                    ]
                )
                self.assertEqual(
                    [token.kind for token in tokens[2:]],
                    [TOKEN_KIND_CANT, TOKEN_KIND_MOVE, TOKEN_KIND_MOVE],
                )
                cant, click, execution = tokens[2:]
                self.assertEqual(cant.action, "slp")
                self.assertEqual(click.action, "sleeptalk")
                self.assertFalse(click.called)
                self.assertEqual(execution.action, "bodyslam")
                self.assertTrue(execution.called)
                self.assertAlmostEqual(execution.damage_fraction, 0.30)

    def test_rest_talk_turn_is_one_decision_opportunity(self) -> None:
        # Three tokens, ONE controllable decision: the opportunity counter must not
        # inflate the switch-tendency denominator on RestTalk turns.
        lines = _leads("Skarmory", "Snorlax") + [
            "|cant|p2a: Snorlax|slp",
            "|move|p2a: Snorlax|Sleep Talk|p2a: Snorlax",
            "|move|p2a: Snorlax|Body Slam|p1a: Skarmory|[from] Sleep Talk",
            "|-damage|p1a: Skarmory|70/100",
            "|upkeep",
            "|turn|2",
        ]
        stats = extract_tendency_stats(parse_showdown_replay(lines), perspective_slot="p1")
        self.assertEqual(stats.opponent_decision_opportunities, 1)

    def test_cant_emits_token_with_reason(self) -> None:
        tokens = _tokens(
            _leads("Snorlax", "Skarmory")
            + [
                "|cant|p1a: Snorlax|slp",
                "|upkeep",
                "|turn|2",
            ]
        )
        cant = tokens[2]
        self.assertEqual(cant.kind, TOKEN_KIND_CANT)
        self.assertEqual(cant.action, "slp")
        self.assertEqual(cant.actor_species, "Snorlax")


class TransformTest(unittest.TestCase):
    def test_transformed_bit_with_base_species_attribution(self) -> None:
        tokens = _tokens(
            _leads("Ditto", "Heracross")
            + [
                "|move|p1a: Ditto|Transform|p2a: Heracross",
                "|-transform|p1a: Ditto|p2a: Heracross",
                "|upkeep",
                "|turn|2",
                "|move|p1a: Ditto|Megahorn|p2a: Heracross",
                "|-damage|p2a: Heracross|40/100",
                "|upkeep",
                "|turn|3",
                "|switch|p1a: Zapdos|Zapdos, L75|100/100",
                "|upkeep",
                "|turn|4",
                "|switch|p1a: Ditto|Ditto, L80|100/100",
                "|upkeep",
                "|turn|5",
                "|move|p1a: Ditto|Tackle|p2a: Heracross",
                "|-damage|p2a: Heracross|35/100",
                "|upkeep",
                "|turn|6",
            ]
        )
        transform, megahorn, tackle = _moves_only(tokens)
        self.assertFalse(transform.transformed)  # not yet transformed when declared
        # Copied-move usage is flagged and stays attributed to slot + BASE species.
        self.assertTrue(megahorn.transformed)
        self.assertEqual(megahorn.actor_slot, "p1")
        self.assertEqual(megahorn.actor_species, "Ditto")
        self.assertEqual(megahorn.action, "megahorn")
        # Switching out ends the transform instance.
        self.assertFalse(tackle.transformed)


class PursuitInterceptTest(unittest.TestCase):
    def test_intercept_flagged_by_activate_marker(self) -> None:
        tokens = _tokens(
            _leads()
            + [
                "|-activate|p2a: Alakazam|move: Pursuit",
                "|move|p1a: Tyranitar|Pursuit|p2a: Alakazam",
                "|-damage|p2a: Alakazam|20/100",
                "|switch|p2a: Starmie|Starmie, L76|100/100",
                "|upkeep",
                "|turn|2",
            ]
        )
        pursuit = _moves_only(tokens)[0]
        self.assertTrue(pursuit.pursuit_intercept)
        # The intercepted (declared) switch still emits its own token.
        self.assertEqual(tokens[3].kind, TOKEN_KIND_SWITCH)
        self.assertEqual(tokens[3].action, "Starmie")

    def test_no_intercept_when_target_stays(self) -> None:
        tokens = _tokens(
            _leads()
            + [
                "|move|p1a: Tyranitar|Pursuit|p2a: Alakazam",
                "|-damage|p2a: Alakazam|70/100",
                "|move|p2a: Alakazam|Psychic|p1a: Tyranitar",
                "|-damage|p1a: Tyranitar|60/100",
                "|upkeep",
                "|turn|2",
            ]
        )
        pursuit = _moves_only(tokens)[0]
        self.assertFalse(pursuit.pursuit_intercept)

    def test_no_intercept_on_plain_ko_with_real_engine_ordering(self) -> None:
        # Adversarial case from review: the REAL engine places faint-replacements
        # BEFORE |upkeep| (faint -> | -> switch -> residuals -> upkeep). A plain
        # Pursuit KO (no -activate marker) must NOT flag, and must NOT count as a
        # switch-predict observation.
        lines = _leads() + [
            "|move|p1a: Tyranitar|Pursuit|p2a: Alakazam",
            "|-damage|p2a: Alakazam|0 fnt",
            "|faint|p2a: Alakazam",
            "|",
            "|switch|p2a: Starmie|Starmie, L76|100/100",
            "|",
            "|-heal|p1a: Tyranitar|100/100|[from] item: Leftovers",
            "|upkeep",
            "|turn|2",
        ]
        replay = parse_showdown_replay(lines)
        tokens = extract_transition_tokens(replay, perspective_slot="p1")
        pursuit = _moves_only(tokens)[0]
        self.assertFalse(pursuit.pursuit_intercept)
        self.assertTrue(pursuit.ko)
        stats = extract_tendency_stats(replay, perspective_slot="p2")
        self.assertEqual(stats.pursuit_intercept_predict_count, 0)

    def test_intercept_through_substitute(self) -> None:
        # Adversarial case from review: an intercepted switch by a mon behind a sub
        # produces no untagged -damage — the marker must still flag the intercept.
        tokens = _tokens(
            _leads()
            + [
                "|-activate|p2a: Alakazam|move: Pursuit",
                "|move|p1a: Tyranitar|Pursuit|p2a: Alakazam",
                "|-activate|p2a: Alakazam|Substitute|[damage]",
                "|switch|p2a: Starmie|Starmie, L76|100/100",
                "|upkeep",
                "|turn|2",
            ]
        )
        pursuit = _moves_only(tokens)[0]
        self.assertTrue(pursuit.pursuit_intercept)
        self.assertEqual(pursuit.damage_outcome, DAMAGE_OUTCOME_HIT_SUB)
        self.assertEqual(pursuit.damage_fraction, 0.0)

    def test_ko_intercept_flagged_via_marker(self) -> None:
        # KO during interception: the marker makes detection exact regardless of how
        # the declared switch completes (completion semantics stay the open experiment).
        tokens = _tokens(
            _leads()
            + [
                "|-activate|p2a: Alakazam|move: Pursuit",
                "|move|p1a: Tyranitar|Pursuit|p2a: Alakazam",
                "|-damage|p2a: Alakazam|0 fnt",
                "|faint|p2a: Alakazam",
                "|",
                "|switch|p2a: Starmie|Starmie, L76|100/100",
                "|",
                "|upkeep",
                "|turn|2",
            ]
        )
        pursuit = _moves_only(tokens)[0]
        self.assertTrue(pursuit.pursuit_intercept)
        self.assertTrue(pursuit.ko)


class ContextTrioTest(unittest.TestCase):
    def _spikes_weather_lines(self) -> list[str]:
        return _leads("Cloyster", "Kingdra") + [
            "|move|p1a: Cloyster|Spikes|p2a: Kingdra",
            "|-sidestart|p2: Bob|Spikes",
            "|upkeep",
            "|turn|2",
            "|move|p1a: Cloyster|Spikes|p2a: Kingdra",
            "|-sidestart|p2: Bob|Spikes",
            "|move|p2a: Kingdra|Rain Dance|p2a: Kingdra",
            "|-weather|RainDance",
            "|upkeep",
            "|-weather|RainDance|[upkeep]",
            "|turn|3",
            "|move|p2a: Kingdra|Surf|p1a: Cloyster",
            "|-damage|p1a: Cloyster|40/100",
            "|upkeep",
            "|turn|4",
        ]

    def test_trio_reflects_layers_and_weather_at_that_turn(self) -> None:
        tokens = _tokens(self._spikes_weather_lines(), perspective_slot="p1")
        spikes_1, spikes_2, rain_dance, surf = _moves_only(tokens)
        # Captured at declaration time: before the action's own effects land.
        self.assertEqual((spikes_1.own_spikes_layers, spikes_1.opp_spikes_layers), (0, 0))
        self.assertIsNone(spikes_1.weather)
        self.assertEqual((spikes_2.own_spikes_layers, spikes_2.opp_spikes_layers), (0, 1))
        self.assertEqual((rain_dance.own_spikes_layers, rain_dance.opp_spikes_layers), (0, 2))
        self.assertIsNone(rain_dance.weather)
        self.assertEqual((surf.own_spikes_layers, surf.opp_spikes_layers), (0, 2))
        self.assertEqual(surf.weather, "raindance")

    def test_trio_is_perspective_relative(self) -> None:
        tokens = _tokens(self._spikes_weather_lines(), perspective_slot="p2")
        surf = _moves_only(tokens)[3]
        self.assertEqual((surf.own_spikes_layers, surf.opp_spikes_layers), (2, 0))


class SideEffectCategoryTest(unittest.TestCase):
    def _category(self, extra_lines: list[str], leads: list[str] | None = None) -> str:
        tokens = _tokens((leads or _leads()) + extra_lines + ["|upkeep", "|turn|2"])
        return _moves_only(tokens)[-1].side_effect

    def test_status_inflicted(self) -> None:
        self.assertEqual(
            self._category(
                ["|move|p1a: Tyranitar|Thunder Wave|p2a: Alakazam", "|-status|p2a: Alakazam|par"]
            ),
            SIDE_EFFECT_STATUS_INFLICTED,
        )

    def test_hazard_set_and_clear(self) -> None:
        self.assertEqual(
            self._category(["|move|p1a: Tyranitar|Spikes|p2a: Alakazam", "|-sidestart|p2: Bob|Spikes"]),
            SIDE_EFFECT_HAZARD_SET,
        )
        self.assertEqual(
            self._category(
                [
                    "|move|p1a: Tyranitar|Rapid Spin|p2a: Alakazam",
                    "|-damage|p2a: Alakazam|95/100",
                    "|-sideend|p1: Alice|Spikes|[from] move: Rapid Spin|[of] p1a: Tyranitar",
                ]
            ),
            SIDE_EFFECT_HAZARD_CLEAR,
        )

    def test_weather_set(self) -> None:
        self.assertEqual(
            self._category(["|move|p1a: Tyranitar|Sunny Day|p1a: Tyranitar", "|-weather|SunnyDay"]),
            SIDE_EFFECT_WEATHER_SET,
        )

    def test_boost(self) -> None:
        self.assertEqual(
            self._category(["|move|p1a: Tyranitar|Dragon Dance|p1a: Tyranitar", "|-boost|p1a: Tyranitar|atk|1"]),
            SIDE_EFFECT_BOOST,
        )

    def test_drain(self) -> None:
        self.assertEqual(
            self._category(
                [
                    "|move|p1a: Celebi|Giga Drain|p2a: Milotic",
                    "|-damage|p2a: Milotic|80/100",
                    "|-heal|p1a: Celebi|90/100|[from] drain|[of] p2a: Milotic",
                ],
                leads=_leads("Celebi", "Milotic"),
            ),
            SIDE_EFFECT_DRAIN,
        )

    def test_heal(self) -> None:
        self.assertEqual(
            self._category(
                ["|move|p1a: Blissey|Soft-Boiled|p1a: Blissey", "|-heal|p1a: Blissey|100/100"],
                leads=_leads("Blissey", "Milotic"),
            ),
            SIDE_EFFECT_HEAL,
        )

    def test_charging(self) -> None:
        self.assertEqual(
            self._category(
                [
                    "|move|p1a: Venusaur|Solar Beam||[still]",
                    "|-prepare|p1a: Venusaur|Solar Beam",
                ],
                leads=_leads("Venusaur", "Milotic"),
            ),
            SIDE_EFFECT_CHARGING,
        )

    def test_residual_silent_heal_does_not_leak_onto_action_tokens(self) -> None:
        # Adversarial case from review (F3): Leech Seed's recipient heal is [silent]
        # and lands in the residual phase — it must not stamp side_effect=heal onto
        # the actor's unrelated attack token. Real chunk shape from captured game 4.
        lines = _leads("Jumpluff", "Whiscash") + [
            "|move|p2a: Whiscash|Surf|p1a: Jumpluff",
            "|-damage|p1a: Jumpluff|55/100",
            "|",
            "|move|p1a: Jumpluff|Return|p2a: Whiscash",
            "|-damage|p2a: Whiscash|88/100",
            "|",
            "|-damage|p2a: Whiscash|82/100|[from] Leech Seed|[of] p1a: Jumpluff",
            "|-heal|p1a: Jumpluff|63/100|[silent]",
            "|upkeep",
            "|turn|2",
        ]
        surf, return_move = _moves_only(_tokens(lines))
        self.assertEqual(return_move.action, "return")
        self.assertEqual(return_move.side_effect, SIDE_EFFECT_NONE)
        self.assertAlmostEqual(return_move.damage_fraction, 0.12)
        self.assertEqual(surf.side_effect, SIDE_EFFECT_NONE)

    def test_rest_heal_is_silent_and_classifies_as_none(self) -> None:
        # Pinned Tier-1 behavior: Rest's heal is [silent] (excluded by attribution
        # hygiene) and its self-status is not "status-inflicted" -> side_effect none.
        lines = _leads("Snorlax", "Skarmory") + [
            "|move|p1a: Snorlax|Rest|p1a: Snorlax",
            "|-status|p1a: Snorlax|slp|[from] move: Rest",
            "|-heal|p1a: Snorlax|100/100 slp|[silent]",
            "|upkeep",
            "|turn|2",
        ]
        rest = _moves_only(_tokens(lines))[0]
        self.assertEqual(rest.side_effect, SIDE_EFFECT_NONE)


class SelfHpCostTest(unittest.TestCase):
    """SELF_HP_COST: fraction of the ACTOR'S max HP lost to its OWN declared action
    within the action's chunk. Emission shapes engine-verified 2026-07-05 (crash and
    Ghost Curse are bare |-damage| on the actor; Pain Split is tagged -sethp; recoil is
    [from] Recoil; self-faint moves emit no self-damage line at all)."""

    def test_recoil_is_a_cost_and_not_defender_damage(self) -> None:
        tokens = _tokens(
            _leads("Tauros", "Milotic")
            + [
                "|move|p1a: Tauros|Double-Edge|p2a: Milotic",
                "|-damage|p2a: Milotic|60/100",
                "|-damage|p1a: Tauros|87/100|[from] Recoil|[of] p2a: Milotic",
                "|upkeep",
                "|turn|2",
            ]
        )
        move = _moves_only(tokens)[0]
        self.assertAlmostEqual(move.self_hp_cost, 0.13)
        self.assertAlmostEqual(move.damage_fraction, 0.40)

    def test_crash_on_miss_bare_actor_damage(self) -> None:
        # Engine-verified shape: |-miss| then a bare |-damage| on the ACTOR.
        tokens = _tokens(
            _leads("Hitmonlee", "Dugtrio")
            + [
                "|move|p1a: Hitmonlee|High Jump Kick|p2a: Dugtrio|[miss]",
                "|-miss|p1a: Hitmonlee|p2a: Dugtrio",
                "|-damage|p1a: Hitmonlee|62/100",
                "|upkeep",
                "|turn|2",
            ]
        )
        move = _moves_only(tokens)[0]
        self.assertTrue(move.miss)
        self.assertAlmostEqual(move.self_hp_cost, 0.38)
        self.assertEqual(move.damage_fraction, 0.0)

    def test_substitute_and_belly_drum_costs(self) -> None:
        tokens = _tokens(
            _leads("Poliwrath", "Skarmory")
            + [
                "|move|p1a: Poliwrath|Substitute|p1a: Poliwrath",
                "|-start|p1a: Poliwrath|Substitute",
                "|-damage|p1a: Poliwrath|75/100",
                "|upkeep",
                "|turn|2",
                "|move|p1a: Poliwrath|Belly Drum|p1a: Poliwrath",
                "|-damage|p1a: Poliwrath|25/100",
                "|-setboost|p1a: Poliwrath|atk|6|[from] move: Belly Drum",
                "|upkeep",
                "|turn|3",
            ]
        )
        substitute, belly_drum = _moves_only(tokens)
        self.assertAlmostEqual(substitute.self_hp_cost, 0.25)
        # Self-target class: the delta stays visible as damage_fraction too (unchanged
        # pre-existing behavior) — the cost channel is the actor-relative reading.
        self.assertAlmostEqual(substitute.damage_fraction, 0.25)
        self.assertAlmostEqual(belly_drum.self_hp_cost, 0.50)

    def test_ghost_curse_bare_actor_damage(self) -> None:
        tokens = _tokens(
            _leads("Gengar", "Skarmory")
            + [
                "|move|p1a: Gengar|Curse|p2a: Skarmory",
                "|-start|p2a: Skarmory|Curse|[of] p1a: Gengar",
                "|-damage|p1a: Gengar|50/100",
                "|upkeep",
                "|turn|2",
            ]
        )
        curse = _moves_only(tokens)[0]
        self.assertAlmostEqual(curse.self_hp_cost, 0.50)
        self.assertEqual(curse.damage_fraction, 0.0)

    def test_pain_split_costs_only_the_losing_actor(self) -> None:
        lines = _leads("Misdreavus", "Blissey") + [
            # Actor is the loser: healthy user splits with a hurt target.
            "|move|p1a: Misdreavus|Pain Split|p2a: Blissey",
            "|-sethp|p2a: Blissey|60/100|[from] move: Pain Split|[silent]",
            "|-sethp|p1a: Misdreavus|65/100|[from] move: Pain Split",
            "|upkeep",
            "|turn|2",
            # Actor is the gainer: no cost.
            "|move|p1a: Misdreavus|Pain Split|p2a: Blissey",
            "|-sethp|p2a: Blissey|55/100|[from] move: Pain Split|[silent]",
            "|-sethp|p1a: Misdreavus|70/100|[from] move: Pain Split",
            "|upkeep",
            "|turn|3",
        ]
        losing, gaining = _moves_only(_tokens(lines))
        self.assertAlmostEqual(losing.self_hp_cost, 0.35)
        self.assertEqual(gaining.self_hp_cost, 0.0)

    def test_explosion_costs_the_entire_remaining_fraction(self) -> None:
        tokens = _tokens(
            _leads("Golem", "Milotic")
            + [
                "|move|p2a: Milotic|Surf|p1a: Golem",
                "|-damage|p1a: Golem|40/100",
                "|upkeep",
                "|turn|2",
                "|move|p1a: Golem|Explosion|p2a: Milotic",
                "|-damage|p2a: Milotic|10/100",
                "|faint|p1a: Golem",
                "|upkeep",
                "|turn|3",
            ]
        )
        explosion = _moves_only(tokens)[1]
        self.assertEqual(explosion.action, "explosion")
        # Documented choice: the cost is the actor's remaining fraction AT STRIKE.
        self.assertAlmostEqual(explosion.self_hp_cost, 0.40)
        self.assertAlmostEqual(explosion.damage_fraction, 0.90)

    def test_opponent_sourced_and_environmental_tags_are_not_costs(self) -> None:
        # Rough Skin (opponent ability, tagged) and Spikes on a switch-in (tagged,
        # environmental, derivable from the context trio): both excluded.
        tokens = _tokens(
            _leads("Machamp", "Sharpedo")
            + [
                "|move|p1a: Machamp|Cross Chop|p2a: Sharpedo",
                "|-damage|p2a: Sharpedo|40/100",
                "|-damage|p1a: Machamp|88/100|[from] ability: Rough Skin|[of] p2a: Sharpedo",
                "|upkeep",
                "|turn|2",
                "|switch|p1a: Skarmory|Skarmory, L76|100/100",
                "|-damage|p1a: Skarmory|88/100|[from] Spikes",
                "|upkeep",
                "|turn|3",
            ]
        )
        cross_chop = _moves_only(tokens)[0]
        self.assertEqual(cross_chop.self_hp_cost, 0.0)
        switch_in = [t for t in tokens if t.kind == TOKEN_KIND_SWITCH][-1]
        self.assertEqual(switch_in.self_hp_cost, 0.0)

    def test_destiny_bond_faint_in_own_chunk_is_not_a_cost(self) -> None:
        # The attacker faints inside its OWN chunk via the opponent's Destiny Bond —
        # an opponent-set trap, not a cost of the chosen action (whitelist rationale).
        tokens = _tokens(
            _leads("Machamp", "Gengar")
            + [
                "|move|p1a: Machamp|Rock Slide|p2a: Gengar",
                "|-damage|p2a: Gengar|0 fnt",
                "|faint|p2a: Gengar",
                "|-activate|p2a: Gengar|move: Destiny Bond",
                "|faint|p1a: Machamp",
                "|upkeep",
                "|turn|2",
            ]
        )
        rock_slide = _moves_only(tokens)[0]
        self.assertTrue(rock_slide.ko)
        self.assertEqual(rock_slide.self_hp_cost, 0.0)

    def test_chip_and_residual_protections_unchanged(self) -> None:
        tokens = _tokens(
            _leads("Skarmory", "Milotic")
            + [
                "|move|p1a: Skarmory|Drill Peck|p2a: Milotic",
                "|-damage|p2a: Milotic|80/100",
                "|",
                "|-damage|p1a: Skarmory|88/100|[from] brn",
                "|upkeep",
                "|turn|2",
            ]
        )
        drill_peck = _moves_only(tokens)[0]
        self.assertEqual(drill_peck.self_hp_cost, 0.0)


class TendencyStatsTest(unittest.TestCase):
    def _small_game(self) -> list[str]:
        return _leads("Zapdos", "Milotic") + [
            "|move|p1a: Zapdos|Thunderbolt|p2a: Milotic",
            "|-damage|p2a: Milotic|70/100",
            "|move|p2a: Milotic|Surf|p1a: Zapdos",
            "|-damage|p1a: Zapdos|75/100",
            "|upkeep",
            "|turn|2",
            "|switch|p2a: Blissey|Blissey, L68|100/100",
            "|move|p1a: Zapdos|Thunderbolt|p2a: Blissey",
            "|-damage|p2a: Blissey|85/100",
            "|upkeep",
            "|turn|3",
            "|move|p1a: Zapdos|Thunderbolt|p2a: Blissey",
            "|-damage|p2a: Blissey|70/100",
            "|move|p2a: Blissey|Soft-Boiled|p2a: Blissey",
            "|-heal|p2a: Blissey|100/100",
            "|upkeep",
            "|turn|4",
        ]

    def test_global_switch_tendency_pair(self) -> None:
        replay = parse_showdown_replay(self._small_game())
        stats = extract_tendency_stats(replay, perspective_slot="p1")
        self.assertEqual(stats.perspective_slot, "p1")
        self.assertEqual(stats.opponent_slot, "p2")
        # p2's decisions: turn-1 move, turn-2 voluntary switch, turn-3 move.
        self.assertEqual(stats.opponent_switch_count, 1)
        self.assertEqual(stats.opponent_decision_opportunities, 3)
        # From the other seat, p1 never switched.
        mirrored = extract_tendency_stats(replay, perspective_slot="p2")
        self.assertEqual(mirrored.opponent_switch_count, 0)
        self.assertEqual(mirrored.opponent_decision_opportunities, 3)
        self.assertEqual(mirrored.my_switch_turn_count, 1)

    def test_per_opponent_mon_triples(self) -> None:
        replay = parse_showdown_replay(self._small_game())
        stats = extract_tendency_stats(replay, perspective_slot="p1")
        by_species = {entry.species: entry for entry in stats.opponent_mon_tendencies}
        self.assertEqual(set(by_species), {"Milotic", "Blissey"})
        milotic = by_species["Milotic"]
        self.assertEqual(milotic.slot, "p2")
        self.assertEqual(milotic.stayed_and_attacked, 1)
        self.assertEqual(milotic.switched_out_before_attacking, 0)
        self.assertEqual(milotic.turns_active, 2)  # active at the |turn|1 and |turn|2 marks
        blissey = by_species["Blissey"]
        self.assertEqual(blissey.stayed_and_attacked, 1)
        self.assertEqual(blissey.turns_active, 2)

    def test_switched_out_before_attacking(self) -> None:
        lines = _leads("Zapdos", "Milotic") + [
            "|move|p1a: Zapdos|Thunderbolt|p2a: Milotic",
            "|-damage|p2a: Milotic|70/100",
            "|upkeep",
            "|turn|2",
            "|switch|p2a: Blissey|Blissey, L68|100/100",
            "|upkeep",
            "|turn|3",
        ]
        stats = extract_tendency_stats(parse_showdown_replay(lines), perspective_slot="p1")
        by_species = {entry.species: entry for entry in stats.opponent_mon_tendencies}
        self.assertEqual(by_species["Milotic"].switched_out_before_attacking, 1)
        self.assertEqual(by_species["Milotic"].stayed_and_attacked, 0)

    def test_weather_reveal_source_split(self) -> None:
        lines = _leads("Tyranitar", "Kingdra") + [
            "|move|p2a: Kingdra|Rain Dance|p2a: Kingdra",
            "|-weather|RainDance",
            "|upkeep",
            "|turn|2",
        ]
        stats = extract_tendency_stats(parse_showdown_replay(lines), perspective_slot="p1")
        self.assertEqual(len(stats.opponent_weather_reveals), 1)
        reveal = stats.opponent_weather_reveals[0]
        self.assertEqual(reveal.weather, "raindance")
        self.assertFalse(reveal.from_ability)
        # Ability weather on switch-in: permanent, double reveal.
        ability_lines = [
            "|player|p1|Alice|",
            "|player|p2|Bob|",
            "|switch|p1a: Zapdos|Zapdos, L75|100/100",
            "|switch|p2a: Tyranitar|Tyranitar, L74|100/100",
            "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Tyranitar",
            "|turn|1",
        ]
        ability_stats = extract_tendency_stats(
            parse_showdown_replay(ability_lines), perspective_slot="p1"
        )
        self.assertEqual(
            ability_stats.opponent_weather_reveals,
            (type(reveal)(weather="sandstorm", from_ability=True),),
        )

    def test_solar_beam_release_turn_is_not_an_opportunity(self) -> None:
        # Review F5: the release of a two-turn charge is locked — no stay-or-switch
        # decision. The charge turn counts; the release turn contributes zero.
        lines = _leads("Skarmory", "Venusaur") + [
            "|move|p2a: Venusaur|Solar Beam||[still]",
            "|-prepare|p2a: Venusaur|Solar Beam",
            "|upkeep",
            "|turn|2",
            "|move|p2a: Venusaur|Solar Beam|p1a: Skarmory",
            "|-damage|p1a: Skarmory|60/100",
            "|upkeep",
            "|turn|3",
        ]
        replay = parse_showdown_replay(lines)
        tokens = extract_transition_tokens(replay, perspective_slot="p1")
        charge, release = [t for t in tokens if t.actor_slot == "p2" and t.kind == TOKEN_KIND_MOVE]
        self.assertEqual(charge.side_effect, SIDE_EFFECT_CHARGING)
        self.assertAlmostEqual(release.damage_fraction, 0.40)
        stats = extract_tendency_stats(replay, perspective_slot="p1")
        self.assertEqual(stats.opponent_decision_opportunities, 1)

    def test_prediction_channel_tier1_inputs(self) -> None:
        lines = _leads("Machamp", "Blissey") + [
            "|move|p2a: Blissey|Protect|p2a: Blissey",
            "|-singleturn|p2a: Blissey|Protect",
            "|move|p1a: Machamp|Cross Chop|p2a: Blissey",
            "|-activate|p2a: Blissey|move: Protect",
            "|upkeep",
            "|turn|2",
            "|-activate|p1a: Machamp|move: Pursuit",
            "|move|p2a: Blissey|Pursuit|p1a: Machamp",
            "|-damage|p1a: Machamp|80/100",
            "|switch|p1a: Starmie|Starmie, L76|100/100",
            "|upkeep",
            "|turn|3",
        ]
        stats = extract_tendency_stats(parse_showdown_replay(lines), perspective_slot="p1")
        self.assertEqual(stats.blocked_on_our_attack_count, 1)
        self.assertEqual(stats.pursuit_intercept_predict_count, 1)
        self.assertEqual(stats.my_switch_turn_count, 1)


class FixtureReplayTest(unittest.TestCase):
    def test_p2_seat_replay_tokens(self) -> None:
        replay = parse_showdown_replay(
            fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1"
        )
        tokens = extract_transition_tokens(replay, perspective_slot="p2")
        # Three switch send-outs + two declared moves; |player|/|request| lines emit nothing.
        self.assertEqual(len(tokens), 5)
        self.assertEqual(
            [token.kind for token in tokens],
            [TOKEN_KIND_SWITCH] * 3 + [TOKEN_KIND_MOVE] * 2,
        )
        arcanine, xatu, charizard, flamethrower, psychic = tokens
        self.assertEqual((arcanine.actor_slot, arcanine.action), ("p1", "Arcanine"))
        self.assertEqual((charizard.actor_slot, charizard.action), ("p2", "Charizard"))
        self.assertEqual(flamethrower.actor_species, "Charizard")
        self.assertEqual(flamethrower.action, "flamethrower")
        self.assertAlmostEqual(flamethrower.damage_fraction, 0.30)
        self.assertEqual(flamethrower.damage_outcome, DAMAGE_OUTCOME_NORMAL)
        self.assertFalse(flamethrower.ko)
        self.assertIsNone(flamethrower.weather)
        self.assertEqual(psychic.actor_species, "Xatu")
        self.assertAlmostEqual(psychic.damage_fraction, 0.20)

    def test_p2_seat_replay_tendencies(self) -> None:
        replay = parse_showdown_replay(
            fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1"
        )
        stats = extract_tendency_stats(replay, perspective_slot="p2")
        # p1's Xatu switch (Arcanine out before attacking) is the one voluntary switch.
        # The fixture has no |turn| lines, so the switch and Xatu's Psychic share the
        # turn-0 bucket: opportunities are per side per turn, hence 1.
        self.assertEqual(stats.opponent_switch_count, 1)
        self.assertEqual(stats.opponent_decision_opportunities, 1)
        by_species = {entry.species: entry for entry in stats.opponent_mon_tendencies}
        self.assertEqual(by_species["Arcanine"].switched_out_before_attacking, 1)
        self.assertEqual(by_species["Xatu"].stayed_and_attacked, 1)
        self.assertEqual(stats.my_switch_turn_count, 0)


class ConfusionSelfHitTest(unittest.TestCase):
    """Spec v3 change 10: the SLOWER confused mon's untagged self-hit folds into the
    opponent's still-open move window's damage_fraction (kept frozen for v2.2), and the
    fold records the self-hit's own fraction additively so a v3 encode can subtract it."""

    # The exact reproduction from the change brief: p2 Surf (0.17) on p1, then p1 (slower,
    # confused) self-hits for 0.10 with an UNTAGGED -damage and no |move|/|cant| line.
    _CONFUSION_LINES = _leads() + [
        "|move|p2a: Alakazam|Surf|p1a: Tyranitar",
        "|-damage|p1a: Tyranitar|83/100",  # Surf = 0.17
        "|-activate|p1a: Tyranitar|confusion",
        "|-damage|p1a: Tyranitar|73/100",  # self-hit = 0.10 (untagged)
        "|upkeep",
        "|turn|2",
    ]

    def _surf(self, lines):
        surfs = [t for t in _moves_only(_tokens(lines)) if t.action == "surf"]
        self.assertEqual(len(surfs), 1)
        return surfs[0]

    def test_self_hit_folds_into_damage_fraction_but_is_recorded_separately(self) -> None:
        surf = self._surf(self._CONFUSION_LINES)
        # The v2.2 field is FROZEN: the self-hit is still folded into damage_fraction.
        self.assertAlmostEqual(surf.damage_fraction, 0.27)
        # The additive v3 metadata isolates the self-hit's own fraction + a presence flag.
        self.assertTrue(surf.confusion_selfhit)
        self.assertAlmostEqual(surf.confusion_selfhit_fraction, 0.10)
        # The move's own damage (self-hit removed) is the corrected value a v3 encode writes.
        self.assertAlmostEqual(
            surf.damage_fraction - surf.confusion_selfhit_fraction, 0.17
        )

    def test_no_confusion_leaves_the_fields_default(self) -> None:
        # Identical log minus the confusion self-hit: damage_fraction is just Surf's 0.17
        # and neither v3 field is engaged.
        clean = _leads() + [
            "|move|p2a: Alakazam|Surf|p1a: Tyranitar",
            "|-damage|p1a: Tyranitar|83/100",
            "|upkeep",
            "|turn|2",
        ]
        surf = self._surf(clean)
        self.assertAlmostEqual(surf.damage_fraction, 0.17)
        self.assertFalse(surf.confusion_selfhit)
        self.assertAlmostEqual(surf.confusion_selfhit_fraction, 0.0)

    def test_confused_mon_that_still_moves_does_not_arm_the_latch(self) -> None:
        # |-activate|confusion followed by a real |move| (the mon shook off confusion and
        # attacked): the latch is cleared by the move, so the opponent's window is untouched.
        lines = _leads() + [
            "|move|p2a: Alakazam|Surf|p1a: Tyranitar",
            "|-damage|p1a: Tyranitar|83/100",
            "|-activate|p1a: Tyranitar|confusion",
            "|move|p1a: Tyranitar|Crunch|p2a: Alakazam",
            "|-damage|p2a: Alakazam|60/100",
            "|upkeep",
            "|turn|2",
        ]
        surf = self._surf(lines)
        self.assertAlmostEqual(surf.damage_fraction, 0.17)
        self.assertFalse(surf.confusion_selfhit)

    def test_tagged_confusion_self_hit_does_not_pollute_and_is_not_recorded(self) -> None:
        # A [from] confusion-tagged self-hit never folds into damage_fraction (no bug to
        # correct), so neither v3 field engages.
        lines = _leads() + [
            "|move|p2a: Alakazam|Surf|p1a: Tyranitar",
            "|-damage|p1a: Tyranitar|83/100",
            "|-activate|p1a: Tyranitar|confusion",
            "|-damage|p1a: Tyranitar|73/100|[from] confusion",
            "|upkeep",
            "|turn|2",
        ]
        surf = self._surf(lines)
        self.assertAlmostEqual(surf.damage_fraction, 0.17)
        self.assertFalse(surf.confusion_selfhit)
        self.assertAlmostEqual(surf.confusion_selfhit_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
