"""Turn-merged transition tokens (v2.1 batch 3): merge semantics + bijection.

Synthetic protocol shapes here mirror ENGINE-VERIFIED emission sequences captured from
the vendored gen3 Showdown on 2026-07-05 (see test_turn_merged_engine.py for the live
counterparts): Explosion double-faint cold pair, hazard-sack fizzle (the opponent's
declared action — targeted or not — emits NO protocol line), Pursuit KO-intercept
switch continuation (engine hint: "Previously chosen switches continue in Gen 2-4"),
Baton Pass mid-turn completion, and sequential (non-cold) same-turn replacements.
"""

from pathlib import Path
import unittest

from pokezero.showdown import parse_showdown_replay
from pokezero.transitions import (
    DAMAGE_OUTCOME_HIT_SUB,
    SIDE_EFFECT_CHARGING,
    SIDE_EFFECT_HAZARD_SET,
    SIDE_EFFECT_WEATHER_SET,
    TOKEN_KIND_CANT,
    TOKEN_KIND_MOVE,
    TOKEN_KIND_SWITCH,
    extract_transition_tokens,
)
from pokezero.turn_merged import (
    PHASE_EXTRA,
    PHASE_LEAD,
    PHASE_REPLACEMENT,
    PHASE_TURN,
    SUB_BLOCK_ABSENT,
    SUB_BLOCK_ACTION,
    SUB_BLOCK_NEGATED,
    SUB_BLOCK_PENDING,
    TurnMergedToken,
    extract_turn_merged_tokens,
    flatten_turn_merged_tokens,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "showdown"


def _leads(p1_species: str = "Tyranitar", p2_species: str = "Alakazam") -> list[str]:
    return [
        "|player|p1|Alice|",
        "|player|p2|Bob|",
        f"|switch|p1a: {p1_species}|{p1_species}, L74|100/100",
        f"|switch|p2a: {p2_species}|{p2_species}, L72|100/100",
        "|turn|1",
    ]


def _merged(lines: list[str], perspective_slot: str = "p1"):
    return extract_turn_merged_tokens(
        parse_showdown_replay(lines), perspective_slot=perspective_slot
    )


def _assert_bijection(test: unittest.TestCase, lines: list[str], perspective_slot: str = "p1"):
    replay = parse_showdown_replay(lines)
    merged = extract_turn_merged_tokens(replay, perspective_slot=perspective_slot)
    per_action = extract_transition_tokens(replay, perspective_slot=perspective_slot)
    test.assertEqual(flatten_turn_merged_tokens(merged), per_action)
    return merged


class LeadPhaseTest(unittest.TestCase):
    def test_leads_merge_into_one_pair_token(self) -> None:
        merged = _assert_bijection(self, _leads())
        self.assertEqual(len(merged), 1)
        lead = merged[0]
        self.assertEqual(lead.phase, PHASE_LEAD)
        self.assertEqual(lead.turn, 0)
        self.assertEqual(lead.first.status, SUB_BLOCK_ACTION)
        self.assertEqual(lead.first.actor_slot, "p1")
        self.assertEqual(lead.first.kind, TOKEN_KIND_SWITCH)
        self.assertEqual(lead.first.action, "Tyranitar")
        self.assertEqual(lead.second.actor_slot, "p2")
        self.assertEqual(lead.second.action, "Alakazam")


class NormalTurnTest(unittest.TestCase):
    def test_two_declared_moves_merge_in_speed_order(self) -> None:
        merged = _assert_bijection(
            self,
            _leads()
            + [
                "|move|p2a: Alakazam|Psychic|p1a: Tyranitar",
                "|-damage|p1a: Tyranitar|60/100",
                "|move|p1a: Tyranitar|Rock Slide|p2a: Alakazam",
                "|-supereffective|p2a: Alakazam",
                "|-damage|p2a: Alakazam|10/100",
                "|upkeep",
                "|turn|2",
            ],
        )
        self.assertEqual(len(merged), 2)
        turn = merged[1]
        self.assertEqual(turn.phase, PHASE_TURN)
        # Speed order is explicit: the faster Alakazam is the FIRST sub-block.
        self.assertEqual(turn.first.actor_slot, "p2")
        self.assertEqual(turn.first.action, "psychic")
        self.assertAlmostEqual(turn.first.damage_fraction, 0.40)
        self.assertEqual(turn.second.actor_slot, "p1")
        self.assertEqual(turn.second.action, "rockslide")
        self.assertEqual(turn.second.effectiveness, "super")

    def test_voluntary_switch_plus_move_is_one_turn_token(self) -> None:
        merged = _assert_bijection(
            self,
            _leads()
            + [
                "|switch|p2a: Starmie|Starmie, L76|100/100",
                "|move|p1a: Tyranitar|Rock Slide|p2a: Starmie",
                "|-damage|p2a: Starmie|70/100",
                "|upkeep",
                "|turn|2",
            ],
        )
        self.assertEqual(len(merged), 2)
        turn = merged[1]
        self.assertEqual(turn.first.kind, TOKEN_KIND_SWITCH)  # switches resolve first
        self.assertEqual(turn.first.action, "Starmie")
        self.assertEqual(turn.second.kind, TOKEN_KIND_MOVE)

    def test_cant_is_that_sides_sub_block(self) -> None:
        merged = _assert_bijection(
            self,
            _leads("Snorlax", "Skarmory")
            + [
                "|cant|p1a: Snorlax|slp",
                "|move|p2a: Skarmory|Drill Peck|p1a: Snorlax",
                "|-damage|p1a: Snorlax|80/100",
                "|upkeep",
                "|turn|2",
            ],
        )
        turn = merged[1]
        self.assertEqual(turn.first.kind, TOKEN_KIND_CANT)
        self.assertEqual(turn.first.action, "slp")
        self.assertIsNone(turn.first.cant_reason)  # a bare cant IS the action
        self.assertEqual(turn.second.action, "drillpeck")

    def test_context_trio_stored_once_at_first_declaration(self) -> None:
        # Rain Dance (first mover) changes the weather mid-turn, so this is the
        # documented lossy-trio shape — bijection is exercised on it separately in
        # TrioMergeAllowanceTest; here we pin WHERE the single trio is captured.
        merged = _merged(
            _leads("Cloyster", "Kingdra")
            + [
                "|move|p1a: Cloyster|Spikes|p2a: Kingdra",
                "|-sidestart|p2: Bob|Spikes",
                "|upkeep",
                "|turn|2",
                "|move|p2a: Kingdra|Rain Dance|p2a: Kingdra",
                "|-weather|RainDance",
                "|move|p1a: Cloyster|Spikes|p2a: Kingdra",
                "|-sidestart|p2: Bob|Spikes",
                "|upkeep",
                "|-weather|RainDance|[upkeep]",
                "|turn|3",
            ],
        )
        turn_2 = merged[2]
        self.assertEqual((turn_2.own_spikes_layers, turn_2.opp_spikes_layers), (0, 1))
        self.assertIsNone(turn_2.weather)  # captured before Rain Dance resolved
        self.assertEqual(turn_2.first.side_effect, SIDE_EFFECT_WEATHER_SET)


class RestTalkCollapseTest(unittest.TestCase):
    def test_rest_talk_turn_collapses_to_one_sub_block(self) -> None:
        lines = _leads("Snorlax", "Skarmory") + [
            "|cant|p1a: Snorlax|slp",
            "|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax",
            "|move|p1a: Snorlax|Body Slam|p2a: Skarmory|[from] Sleep Talk",
            "|-damage|p2a: Skarmory|70/100",
            "|move|p2a: Skarmory|Drill Peck|p1a: Snorlax",
            "|-damage|p1a: Snorlax|85/100",
            "|upkeep",
            "|turn|2",
        ]
        merged = _assert_bijection(self, lines)  # three tokens rebuild exactly
        turn = merged[1]
        self.assertEqual(turn.phase, PHASE_TURN)
        sub = turn.first
        self.assertEqual(sub.action, "bodyslam")
        self.assertTrue(sub.called)
        self.assertEqual(sub.cant_reason, "slp")
        self.assertAlmostEqual(sub.damage_fraction, 0.30)
        self.assertEqual(turn.second.action, "drillpeck")

    def test_sleep_talk_click_without_execution_keeps_cant_reason(self) -> None:
        merged = _assert_bijection(
            self,
            _leads("Snorlax", "Skarmory")
            + [
                "|cant|p1a: Snorlax|slp",
                "|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax",
                "|-fail|p1a: Snorlax",
                "|upkeep",
                "|turn|2",
            ],
        )
        sub = merged[1].first
        self.assertEqual(sub.action, "sleeptalk")
        self.assertFalse(sub.called)
        self.assertEqual(sub.cant_reason, "slp")


class BatonPassTest(unittest.TestCase):
    def test_completion_folds_into_the_passers_sub_block(self) -> None:
        # Engine-verified turn shape: BP click -> forceSwitch pause -> completion with
        # [from] Baton Pass -> the slower opponent acts against the NEW mon.
        merged = _assert_bijection(
            self,
            _leads("Jolteon", "Skarmory")
            + [
                "|move|p1a: Jolteon|Baton Pass|p1a: Jolteon",
                "|switch|p1a: Snorlax|Snorlax, L80|100/100|[from] Baton Pass",
                "|move|p2a: Skarmory|Drill Peck|p1a: Snorlax",
                "|-damage|p1a: Snorlax|77/100",
                "|upkeep",
                "|turn|2",
            ],
        )
        self.assertEqual(len(merged), 2)
        turn = merged[1]
        self.assertEqual(turn.first.action, "batonpass")
        self.assertEqual(turn.first.baton_pass_species, "Snorlax")
        self.assertEqual(turn.second.action, "drillpeck")


class NegatedSubBlockTest(unittest.TestCase):
    def test_hazard_sack_fizzle_is_negated_with_mid_turn_replacement(self) -> None:
        # Engine-verified: Shedinja faints to Spikes on switch-in; the opponent's
        # declared action (even a non-targeted Spikes) emits NOTHING; the replacement
        # completes mid-turn and the turn ends.
        lines = _leads("Magikarp", "Skarmory") + [
            "|move|p1a: Magikarp|Splash|p1a: Magikarp",
            "|-nothing",
            "|move|p2a: Skarmory|Spikes|p1a: Magikarp",
            "|-sidestart|p1: Alice|Spikes",
            "|",
            "|upkeep",
            "|turn|2",
            "|switch|p1a: Shedinja|Shedinja, L82|100/100",
            "|-damage|p1a: Shedinja|0 fnt|[from] Spikes",
            "|faint|p1a: Shedinja",
            "|switch|p1a: Sandslash|Sandslash, L80|100/100",
            "|-damage|p1a: Sandslash|88/100|[from] Spikes",
            "|",
            "|upkeep",
            "|turn|3",
        ]
        merged = _assert_bijection(self, lines)
        self.assertEqual([token.phase for token in merged], [PHASE_LEAD, PHASE_TURN, PHASE_TURN, PHASE_REPLACEMENT])
        sack_turn = merged[2]
        self.assertEqual(sack_turn.first.actor_slot, "p1")
        self.assertEqual(sack_turn.first.action, "Shedinja")
        self.assertEqual(sack_turn.first.kind, TOKEN_KIND_SWITCH)
        # The free pivot, made legible: Skarmory's declaration was consumed.
        self.assertEqual(sack_turn.second.status, SUB_BLOCK_NEGATED)
        self.assertEqual(sack_turn.second.actor_slot, "p2")
        self.assertEqual(sack_turn.second.actor_species, "Skarmory")
        # Context trio: one Spikes layer down on the perspective side at declaration.
        self.assertEqual(sack_turn.own_spikes_layers, 1)
        replacement = merged[3]
        self.assertEqual(replacement.phase, PHASE_REPLACEMENT)
        self.assertEqual(replacement.first.action, "Sandslash")
        self.assertEqual(replacement.second.status, SUB_BLOCK_ABSENT)

    def test_faster_ko_negates_the_victims_declared_action(self) -> None:
        merged = _assert_bijection(
            self,
            _leads("Alakazam", "Machamp")
            + [
                "|move|p1a: Alakazam|Psychic|p2a: Machamp",
                "|-damage|p2a: Machamp|0 fnt",
                "|faint|p2a: Machamp",
                "|",
                "|switch|p2a: Starmie|Starmie, L76|100/100",
                "|",
                "|upkeep",
                "|turn|2",
            ],
        )
        turn = merged[1]
        self.assertTrue(turn.first.ko)
        self.assertEqual(turn.second.status, SUB_BLOCK_NEGATED)
        self.assertEqual(turn.second.actor_species, "Machamp")


class PendingSubBlockTest(unittest.TestCase):
    """Review MED-1: NEGATED requires PROOF of consumption (turn closed or a mid-turn
    faint). A replay prefix cut at a mid-turn forceSwitch boundary — the Baton Pass
    completion choice, a REAL live decision point — must read the opponent's
    still-pending action as PENDING, never as the free-pivot negation."""

    _BP_PREFIX = None  # set in setUp for clarity

    def setUp(self) -> None:
        self._BP_PREFIX = _leads("Jolteon", "Skarmory") + [
            "|move|p1a: Jolteon|Baton Pass|p1a: Jolteon",
        ]

    def test_bp_completion_boundary_is_pending_not_negated(self) -> None:
        merged = _merged(self._BP_PREFIX)
        turn = merged[1]
        self.assertEqual(turn.phase, PHASE_TURN)
        self.assertEqual(turn.first.action, "batonpass")
        self.assertIsNone(turn.first.baton_pass_species)  # completion not chosen yet
        self.assertEqual(turn.second.status, SUB_BLOCK_PENDING)
        self.assertEqual(turn.second.actor_slot, "p2")
        self.assertEqual(turn.second.actor_species, "Skarmory")
        # Flatten skips pending halves; bijection holds on the prefix.
        _assert_bijection(self, self._BP_PREFIX)

    def test_pending_resolves_to_the_executed_action_after_completion(self) -> None:
        completed = self._BP_PREFIX + [
            "|switch|p1a: Snorlax|Snorlax, L80|100/100|[from] Baton Pass",
            "|move|p2a: Skarmory|Drill Peck|p1a: Snorlax",
            "|-damage|p1a: Snorlax|77/100",
            "|upkeep",
            "|turn|2",
        ]
        merged = _assert_bijection(self, completed)
        turn = merged[1]
        self.assertEqual(turn.first.baton_pass_species, "Snorlax")
        self.assertEqual(turn.second.status, SUB_BLOCK_ACTION)
        self.assertEqual(turn.second.action, "drillpeck")

    def test_mid_turn_faint_confirms_negation_before_the_turn_closes(self) -> None:
        # The hazard-sack pause boundary: the faint IS the consumption proof
        # (engine-verified full cancel), so NEGATED fires even though the turn is open.
        prefix = _leads("Magikarp", "Skarmory") + [
            "|move|p1a: Magikarp|Splash|p1a: Magikarp",
            "|-nothing",
            "|move|p2a: Skarmory|Spikes|p1a: Magikarp",
            "|-sidestart|p1: Alice|Spikes",
            "|",
            "|upkeep",
            "|turn|2",
            "|switch|p1a: Shedinja|Shedinja, L82|100/100",
            "|-damage|p1a: Shedinja|0 fnt|[from] Spikes",
            "|faint|p1a: Shedinja",
        ]
        merged = _merged(prefix)
        sack_turn = merged[2]
        self.assertEqual(sack_turn.second.status, SUB_BLOCK_NEGATED)
        self.assertEqual(sack_turn.second.actor_species, "Skarmory")

    def test_open_turn_after_first_mover_without_faint_is_pending(self) -> None:
        prefix = _leads("Alakazam", "Machamp") + [
            "|move|p1a: Alakazam|Psychic|p2a: Machamp",
            "|-damage|p2a: Machamp|55/100",
        ]
        merged = _merged(prefix)
        self.assertEqual(merged[1].second.status, SUB_BLOCK_PENDING)


class ReplacementPhaseTest(unittest.TestCase):
    def _explosion_lines(self) -> list[str]:
        # Engine-verified cold pair: both faints, then both replacements back-to-back
        # in ONE forceSwitch cycle, before |upkeep|.
        return _leads("Golem", "Abra") + [
            "|move|p1a: Golem|Explosion|p2a: Abra",
            "|-damage|p2a: Abra|0 fnt",
            "|faint|p1a: Golem",
            "|faint|p2a: Abra",
            "|",
            "|switch|p1a: Sandslash|Sandslash, L80|100/100",
            "|switch|p2a: Starmie|Starmie, L76|100/100",
            "|",
            "|upkeep",
            "|turn|2",
        ]

    def test_explosion_double_faint_merges_into_a_cold_pair(self) -> None:
        merged = _assert_bijection(self, self._explosion_lines())
        self.assertEqual([token.phase for token in merged], [PHASE_LEAD, PHASE_TURN, PHASE_REPLACEMENT])
        explosion_turn = merged[1]
        self.assertEqual(explosion_turn.first.action, "explosion")
        self.assertTrue(explosion_turn.first.ko)
        self.assertEqual(explosion_turn.second.status, SUB_BLOCK_NEGATED)
        pair = merged[2]
        self.assertEqual(pair.first.status, SUB_BLOCK_ACTION)
        self.assertEqual(pair.second.status, SUB_BLOCK_ACTION)
        self.assertEqual(pair.first.action, "Sandslash")  # engine emission order
        self.assertEqual(pair.second.action, "Starmie")
        self.assertEqual(pair.turn, 1)

    def test_cold_pair_keeps_engine_emission_order_when_reversed(self) -> None:
        lines = self._explosion_lines()
        lines[10], lines[11] = lines[11], lines[10]  # p2's replacement emitted first
        merged = _assert_bijection(self, lines)
        pair = merged[2]
        self.assertEqual(pair.first.actor_slot, "p2")
        self.assertEqual(pair.second.actor_slot, "p1")

    def test_sequential_same_turn_replacements_stay_single_tokens(self) -> None:
        # Engine-verified: move KO -> p2 replaces mid-turn (p1 still alive); THEN p1
        # faints to residual poison -> second, separate forceSwitch cycle after upkeep.
        lines = _leads("Machamp", "Abra") + [
            "|move|p1a: Machamp|Cross Chop|p2a: Abra",
            "|-damage|p2a: Abra|0 fnt",
            "|faint|p2a: Abra",
            "|switch|p2a: Starmie|Starmie, L76|100/100",
            "|",
            "|-damage|p1a: Machamp|0 fnt|[from] psn",
            "|faint|p1a: Machamp",
            "|upkeep",
            "|switch|p1a: Sandslash|Sandslash, L80|100/100",
            "|turn|2",
        ]
        merged = _assert_bijection(self, lines)
        self.assertEqual(
            [token.phase for token in merged],
            [PHASE_LEAD, PHASE_TURN, PHASE_REPLACEMENT, PHASE_REPLACEMENT],
        )
        first_replacement, second_replacement = merged[2], merged[3]
        self.assertEqual(first_replacement.first.actor_slot, "p2")
        self.assertEqual(first_replacement.second.status, SUB_BLOCK_ABSENT)
        self.assertEqual(second_replacement.first.actor_slot, "p1")
        self.assertEqual(second_replacement.second.status, SUB_BLOCK_ABSENT)
        # Post-upkeep replacement still belongs to the turn of the faint.
        self.assertEqual(second_replacement.turn, 1)


class PursuitTest(unittest.TestCase):
    def test_intercepting_pursuit_is_first_and_the_switch_completes_second(self) -> None:
        merged = _assert_bijection(
            self,
            _leads()
            + [
                "|-activate|p2a: Alakazam|move: Pursuit",
                "|move|p1a: Tyranitar|Pursuit|p2a: Alakazam|[from]Pursuit",
                "|-damage|p2a: Alakazam|20/100",
                "|switch|p2a: Starmie|Starmie, L76|100/100",
                "|upkeep",
                "|turn|2",
            ],
        )
        turn = merged[1]
        self.assertEqual(len(merged), 2)
        self.assertTrue(turn.first.pursuit_intercept)
        self.assertEqual(turn.second.kind, TOKEN_KIND_SWITCH)
        self.assertEqual(turn.second.action, "Starmie")

    def test_pursuit_ko_continuation_is_the_declared_switch_not_a_replacement(self) -> None:
        # Engine-verified: "Previously chosen switches continue in Gen 2-4 after a
        # Pursuit target faints" — the switch completes in the same breath, no
        # forceSwitch cycle.
        merged = _assert_bijection(
            self,
            _leads()
            + [
                "|-activate|p2a: Alakazam|move: Pursuit",
                "|move|p1a: Tyranitar|Pursuit|p2a: Alakazam|[from]Pursuit",
                "|-damage|p2a: Alakazam|0 fnt",
                "|faint|p2a: Alakazam",
                "|switch|p2a: Starmie|Starmie, L76|100/100",
                "|",
                "|upkeep",
                "|turn|2",
            ],
        )
        self.assertEqual([token.phase for token in merged], [PHASE_LEAD, PHASE_TURN])
        turn = merged[1]
        self.assertTrue(turn.first.pursuit_intercept)
        self.assertTrue(turn.first.ko)
        self.assertEqual(turn.second.status, SUB_BLOCK_ACTION)
        self.assertEqual(turn.second.kind, TOKEN_KIND_SWITCH)
        self.assertEqual(turn.second.action, "Starmie")

    def test_plain_pursuit_ko_replacement_stays_a_replacement(self) -> None:
        # No -activate marker: the target was NOT switching; its replacement is a real
        # (fresh-choice) replacement phase.
        merged = _assert_bijection(
            self,
            _leads()
            + [
                "|move|p1a: Tyranitar|Pursuit|p2a: Alakazam",
                "|-damage|p2a: Alakazam|0 fnt",
                "|faint|p2a: Alakazam",
                "|",
                "|switch|p2a: Starmie|Starmie, L76|100/100",
                "|",
                "|upkeep",
                "|turn|2",
            ],
        )
        self.assertEqual(
            [token.phase for token in merged], [PHASE_LEAD, PHASE_TURN, PHASE_REPLACEMENT]
        )
        self.assertEqual(merged[1].second.status, SUB_BLOCK_NEGATED)


class OdditySweepTest(unittest.TestCase):
    def test_drag_emits_no_sub_block_and_negated_species_survives_the_drag(self) -> None:
        # Roar's forced |drag| is not a declared action; the NEXT turn's negated
        # species lookup must reflect the dragged-in mon (drag-safe occupant map).
        lines = _leads("Skarmory", "Milotic") + [
            "|move|p2a: Milotic|Surf|p1a: Skarmory",
            "|-damage|p1a: Skarmory|70/100",
            "|move|p1a: Skarmory|Roar|p2a: Milotic",
            "|drag|p2a: Blissey|Blissey, L68|100/100",
            "|upkeep",
            "|turn|2",
            "|move|p1a: Skarmory|Drill Peck|p2a: Blissey",
            "|-damage|p2a: Blissey|0 fnt",
            "|faint|p2a: Blissey",
            "|",
            "|upkeep",
            "|turn|3",
        ]
        merged = _assert_bijection(self, lines)
        roar_turn = merged[1]
        self.assertEqual(roar_turn.phase, PHASE_TURN)
        self.assertEqual(roar_turn.first.action, "surf")
        self.assertEqual(roar_turn.second.action, "roar")
        ko_turn = merged[2]
        self.assertEqual(ko_turn.second.status, SUB_BLOCK_NEGATED)
        self.assertEqual(ko_turn.second.actor_species, "Blissey")

    def test_charge_release_and_residuals_never_add_tokens(self) -> None:
        merged = _assert_bijection(
            self,
            _leads("Venusaur", "Blissey")
            + [
                "|move|p1a: Venusaur|Solar Beam||[still]",
                "|-prepare|p1a: Venusaur|Solar Beam",
                "|move|p2a: Blissey|Soft-Boiled|p2a: Blissey",
                "|-heal|p2a: Blissey|100/100",
                "|",
                "|-heal|p1a: Venusaur|100/100|[from] item: Leftovers",
                "|-damage|p2a: Blissey|94/100|[from] Leech Seed|[of] p1a: Venusaur",
                "|-heal|p1a: Venusaur|100/100|[silent]",
                "|upkeep",
                "|turn|2",
                "|move|p1a: Venusaur|Solar Beam|p2a: Blissey",
                "|-damage|p2a: Blissey|64/100",
                "|upkeep",
                "|turn|3",
            ],
        )
        self.assertEqual([token.phase for token in merged], [PHASE_LEAD, PHASE_TURN, PHASE_TURN])
        self.assertEqual(merged[1].first.side_effect, SIDE_EFFECT_CHARGING)
        release = merged[2]
        self.assertEqual(release.first.action, "solarbeam")
        self.assertEqual(release.second.status, SUB_BLOCK_NEGATED)

    def test_sub_protecting_defender_fields_survive_the_merge(self) -> None:
        merged = _assert_bijection(
            self,
            _leads("Swampert", "Zapdos")
            + [
                "|move|p2a: Zapdos|Substitute|p2a: Zapdos",
                "|-start|p2a: Zapdos|Substitute",
                "|-damage|p2a: Zapdos|75/100",
                "|move|p1a: Swampert|Surf|p2a: Zapdos",
                "|-activate|p2a: Zapdos|Substitute|[damage]",
                "|upkeep",
                "|turn|2",
            ],
        )
        turn = merged[1]
        self.assertEqual(turn.second.damage_outcome, DAMAGE_OUTCOME_HIT_SUB)
        self.assertEqual(turn.second.damage_fraction, 0.0)

    def test_no_extra_phase_on_recognized_shapes(self) -> None:
        lines = _leads("Snorlax", "Skarmory") + [
            "|cant|p1a: Snorlax|slp",
            "|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax",
            "|move|p1a: Snorlax|Body Slam|p2a: Skarmory|[from] Sleep Talk",
            "|-damage|p2a: Skarmory|70/100",
            "|move|p2a: Skarmory|Spikes|p1a: Snorlax",
            "|-sidestart|p1: Alice|Spikes",
            "|upkeep",
            "|turn|2",
        ]
        merged = _assert_bijection(self, lines)
        self.assertNotIn(PHASE_EXTRA, [token.phase for token in merged])


class TrioMergeAllowanceTest(unittest.TestCase):
    def test_second_mover_trio_reconstructs_with_first_movers_context(self) -> None:
        # The ONE documented lossy merge: Spikes lands between the two declarations, so
        # the per-action second-mover token saw (0, 1) while the merged token stores the
        # first mover's (0, 0). The flatten output differs from the per-action stream
        # ONLY in that trio, and only because first.side_effect is hazard-set.
        lines = _leads("Cloyster", "Blissey") + [
            "|move|p1a: Cloyster|Spikes|p2a: Blissey",
            "|-sidestart|p2: Bob|Spikes",
            "|move|p2a: Blissey|Soft-Boiled|p2a: Blissey",
            "|-heal|p2a: Blissey|100/100",
            "|upkeep",
            "|turn|2",
        ]
        replay = parse_showdown_replay(lines)
        merged = extract_turn_merged_tokens(replay, perspective_slot="p1")
        per_action = extract_transition_tokens(replay, perspective_slot="p1")
        flattened = flatten_turn_merged_tokens(merged)
        self.assertEqual(len(flattened), len(per_action))
        mismatches = [
            (rebuilt, original)
            for rebuilt, original in zip(flattened, per_action)
            if rebuilt != original
        ]
        self.assertEqual(len(mismatches), 1)
        rebuilt, original = mismatches[0]
        self.assertEqual(original.action, "softboiled")
        self.assertEqual(original.opp_spikes_layers, 1)  # truth: declared under 1 layer
        self.assertEqual(rebuilt.opp_spikes_layers, 0)  # merged: first mover's trio
        self.assertEqual(
            rebuilt,
            type(rebuilt)(
                **{
                    **original.__dict__,
                    "own_spikes_layers": rebuilt.own_spikes_layers,
                    "opp_spikes_layers": rebuilt.opp_spikes_layers,
                    "weather": rebuilt.weather,
                }
            ),
        )
        self.assertEqual(merged[1].first.side_effect, SIDE_EFFECT_HAZARD_SET)


class PerspectiveTest(unittest.TestCase):
    def test_trio_is_perspective_relative_and_structure_is_not(self) -> None:
        lines = _leads("Cloyster", "Kingdra") + [
            "|move|p1a: Cloyster|Spikes|p2a: Kingdra",
            "|-sidestart|p2: Bob|Spikes",
            "|upkeep",
            "|turn|2",
            "|move|p2a: Kingdra|Surf|p1a: Cloyster",
            "|-damage|p1a: Cloyster|40/100",
            "|move|p1a: Cloyster|Spikes|p2a: Kingdra",
            "|-sidestart|p2: Bob|Spikes",
            "|upkeep",
            "|turn|3",
        ]
        from_p1 = _merged(lines, "p1")
        from_p2 = _merged(lines, "p2")
        self.assertEqual([token.phase for token in from_p1], [token.phase for token in from_p2])
        self.assertEqual((from_p1[2].own_spikes_layers, from_p1[2].opp_spikes_layers), (0, 1))
        self.assertEqual((from_p2[2].own_spikes_layers, from_p2[2].opp_spikes_layers), (1, 0))


class ExplosionFixtureTest(unittest.TestCase):
    """The banked seed-148 engine game (real gen3 randbats, gzipped fixture): a turn-7
    Explosion double-faint with both sides replacing cold — exactly the merged
    replacement-pair shape, from a full engine-generated log rather than synthetic lines."""

    def test_seed148_turn7_is_a_cold_pair_and_the_game_rebuilds(self) -> None:
        import gzip

        raw = gzip.open(FIXTURE_ROOT / "explosion-seed148.log.gz", "rb").read().decode()
        lines = [line for line in raw.split("\n") if line]
        replay = parse_showdown_replay(lines, battle_id="explosion-seed148")
        merged = extract_turn_merged_tokens(replay, perspective_slot="p1")
        per_action = extract_transition_tokens(replay, perspective_slot="p1")

        turn_7 = [token for token in merged if token.turn == 7]
        self.assertEqual(
            [token.phase for token in turn_7], [PHASE_TURN, PHASE_REPLACEMENT]
        )
        explosion_turn, pair = turn_7
        explosion_sub = next(
            sub
            for sub in (explosion_turn.first, explosion_turn.second)
            if sub.status == SUB_BLOCK_ACTION and sub.action == "explosion"
        )
        self.assertEqual(explosion_sub.actor_species, "Weezing")
        self.assertTrue(explosion_sub.ko)
        # The cold pair: both sides replace blind, one merged token.
        self.assertEqual(pair.first.status, SUB_BLOCK_ACTION)
        self.assertEqual(pair.second.status, SUB_BLOCK_ACTION)
        self.assertEqual(pair.first.kind, TOKEN_KIND_SWITCH)
        self.assertEqual(pair.second.kind, TOKEN_KIND_SWITCH)
        self.assertEqual({pair.first.actor_slot, pair.second.actor_slot}, {"p1", "p2"})
        # SELF_HP_COST on the real engine game: Weezing exploded from full HP, so its
        # sub-block carries the entire remaining fraction; the Gligar (defender) side
        # of the token is untouched by the cost channel.
        self.assertAlmostEqual(explosion_sub.self_hp_cost, 1.0)
        gligar_sub = next(
            sub
            for sub in (explosion_turn.first, explosion_turn.second)
            if sub is not explosion_sub and sub.status == SUB_BLOCK_ACTION
        )
        self.assertEqual(gligar_sub.self_hp_cost, 0.0)
        # No other double-faint in the game produced a false pair, no EXTRA fallback
        # fired, and the whole real game reconstructs per-action exactly (this fixture
        # happens to contain no intra-turn trio changer, so the bijection is total).
        self.assertNotIn(PHASE_EXTRA, [token.phase for token in merged])
        self.assertEqual(flatten_turn_merged_tokens(merged), per_action)
        self.assertLess(len(merged), 0.62 * len(per_action))


class ConfusionSelfHitMergeTest(unittest.TestCase):
    """Spec v3 change 10: the confusion self-hit metadata rides the opponent's move
    sub-block and survives the merge/flatten bijection."""

    def test_selfhit_rides_the_first_sub_block_and_survives_bijection(self) -> None:
        lines = _leads() + [
            "|move|p2a: Alakazam|Surf|p1a: Tyranitar",
            "|-damage|p1a: Tyranitar|83/100",
            "|-activate|p1a: Tyranitar|confusion",
            "|-damage|p1a: Tyranitar|73/100",  # slower confused mon self-hits
            "|upkeep",
            "|turn|2",
        ]
        merged = _assert_bijection(self, lines)
        turn = merged[1]
        self.assertEqual(turn.phase, PHASE_TURN)
        # The faster Alakazam is the FIRST sub-block and carries the correction metadata.
        self.assertEqual(turn.first.actor_slot, "p2")
        self.assertEqual(turn.first.action, "surf")
        self.assertAlmostEqual(turn.first.damage_fraction, 0.27)  # frozen v2.2 value
        self.assertTrue(turn.first.confusion_selfhit)
        self.assertAlmostEqual(turn.first.confusion_selfhit_fraction, 0.10)
        # The confused mon's declared action was consumed with no move trace: NEGATED.
        self.assertEqual(turn.second.actor_slot, "p1")
        self.assertEqual(turn.second.status, SUB_BLOCK_NEGATED)


class FixtureReplayTest(unittest.TestCase):
    def test_p2_seat_replay_merges_and_rebuilds(self) -> None:
        lines = (FIXTURE_ROOT / "p2_seat_replay.txt").read_text().splitlines()
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
        merged = extract_turn_merged_tokens(replay, perspective_slot="p2")
        per_action = extract_transition_tokens(replay, perspective_slot="p2")
        self.assertEqual(flatten_turn_merged_tokens(merged), per_action)
        self.assertLess(len(merged), len(per_action))


if __name__ == "__main__":
    unittest.main()
