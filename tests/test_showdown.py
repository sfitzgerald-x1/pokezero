import json
from pathlib import Path
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.belief import CandidateSetSummary
from pokezero.observation import (
    ACTION_CANDIDATE_TOKEN_COUNT,
    FIELD_TOKEN_COUNT,
    OPPONENT_POKEMON_TOKEN_COUNT,
    SELF_POKEMON_TOKEN_COUNT,
)
from pokezero.category_vocab import build_category_vocabulary
from pokezero.dex import MoveInfo, ShowdownDex, SpeciesInfo
from pokezero.showdown import (
    CATEGORY_MOVE_EFFECT,
    CATEGORY_SECONDARY,
    CATEGORY_VOLATILE_OFFSET,
    DEFAULT_REPLAY_OBSERVATION_SPEC,
    NUMERIC_ACTUAL_HP,
    NUMERIC_ACTUAL_SPE,
    NUMERIC_EFFECT_CHANCE,
    NUMERIC_MOVE_PP_FRACTION,
    NUMERIC_OPP_FUTURE_SIGHT,
    NUMERIC_SELF_FUTURE_SIGHT,
    NUMERIC_SELF_HP_COST,
    NUMERIC_TOXIC_STAGE,
    NUMERIC_TURN_COUNT,
    _actual_stats_from_request_row,
    _encode_move_mechanics,
    _max_hp_from_condition,
    _move_pp_fraction,
    NUMERIC_BASE_ATK,
    NUMERIC_BASE_DEF,
    NUMERIC_BASE_HP,
    NUMERIC_BASE_SPA,
    NUMERIC_BASE_SPD,
    NUMERIC_BASE_SPE,
    NUMERIC_BOOST_ATK,
    NUMERIC_BOOST_SPE,
    NUMERIC_LEVEL,
    NUMERIC_OPP_HAZARDS,
    NUMERIC_OPP_SCREENS,
    NUMERIC_SELF_HAZARDS,
    NUMERIC_SELF_SCREENS,
    PlayerRelativePublicEvent,
    _event_detail_category,
    detect_showdown_slot,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
    showdown_choice_for_action,
    showdown_submission_for_action,
)

FIELD_TOKEN_OFFSET = 0

# Shared string→row vocabulary for observation-encoding assertions. Contains every token the
# tests assert on, so each maps to a distinct row; both the encoder and the expected values use
# this same vocab (so assertions hold regardless of exact row numbers).
_TEST_VOCAB = build_category_vocabulary(
    [
        "request_kind:move",
        "species:Charizard", "species:Arcanine", "species:Xatu", "species:Snorlax", "species:Blissey",
        "move:flamethrower", "move:dragonclaw", "move:Flamethrower",
        "event:player", "event:move", "event:-damage",
        "event_actor:self", "event_target:opponent",
        "belief:possible_ability:earlybird", "belief:possible_ability:synchronize",
        "belief:possible_item:leftovers",
        "belief:possible_move:psychic", "belief:possible_move:thunderwave", "belief:possible_move:wish",
        "belief:possible_moves:psychic|thunderwave|wish",
        "weather:raindance",
        "move_effect:brn",
        "volatile:confusion", "volatile:leechseed",
    ]
)


def _phase2_fake_dex() -> ShowdownDex:
    """Minimal dex with the fixture's active self mon (Charizard) + Flamethrower for encoding tests."""
    charizard = SpeciesInfo(
        id="charizard",
        name="Charizard",
        types=("Fire", "Flying"),
        base_stats={"hp": 78, "atk": 84, "def": 78, "spa": 109, "spd": 85, "spe": 100},
    )
    flamethrower = MoveInfo(
        id="flamethrower", name="Flamethrower", type="Fire", category="Special",
        gen3_category="Special", base_power=95, accuracy=100.0, priority=0,
        recoil=False, drain=False, heal=False, status=None, boosts={},
        target="normal", selfdestruct=False, effect_chance=10, effect_label="brn",
    )
    return ShowdownDex(moves={"flamethrower": flamethrower}, species={"charizard": charizard}, type_chart={})


def stable_category_id(value: str) -> int:
    """Test shim: resolve a token string to its row in the shared test vocabulary."""
    return _TEST_VOCAB.encode(value)


class EventDetailCategoryTest(unittest.TestCase):
    def _event(self, event_type: str, primary: str) -> PlayerRelativePublicEvent:
        return PlayerRelativePublicEvent(event_type=event_type, raw_line="", primary=primary)

    def test_enumerable_details_emit_in_vocab_tokens(self) -> None:
        self.assertEqual(_event_detail_category(self._event("move", "Flamethrower")), "move:Flamethrower")
        self.assertEqual(_event_detail_category(self._event("switch", "Snorlax")), "species:Snorlax")
        self.assertEqual(_event_detail_category(self._event("-status", "par")), "status:par")

    def test_unactionable_details_are_dropped(self) -> None:
        # HP strings, usernames, winner identity, and free-form payloads -> None (padding slot).
        self.assertIsNone(_event_detail_category(self._event("-damage", "70/100")))
        self.assertIsNone(_event_detail_category(self._event("-heal", "200/267 tox")))
        self.assertIsNone(_event_detail_category(self._event("player", "SomeUsername")))
        self.assertIsNone(_event_detail_category(self._event("win", "SomeUsername")))
        self.assertIsNone(_event_detail_category(self._event("-weather", "Sandstorm")))


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "showdown"


def fixture_lines(name: str) -> list[str]:
    return (FIXTURE_ROOT / name).read_text(encoding="utf-8").splitlines()


class FakeSetSource:
    def summarize(self, *, format_id, species, revealed_moves, **kwargs):
        if species == "Xatu":
            return CandidateSetSummary(
                species=species,
                candidate_count=2,
                uncertainty=0.5,
                possible_abilities=("Early Bird", "Synchronize"),
                possible_items=("Leftovers",),
                possible_moves=("psychic", "thunderwave", "wish"),
            )
        return CandidateSetSummary(
            species=species,
            candidate_count=1,
            uncertainty=0.25,
            possible_abilities=("Intimidate",),
            possible_items=("Lum Berry", "Leftovers"),
            possible_moves=tuple(revealed_moves),
        )


class ReorderedFakeSetSource(FakeSetSource):
    def summarize(self, *, format_id, species, revealed_moves, **kwargs):
        if species == "Xatu":
            return CandidateSetSummary(
                species=species,
                candidate_count=2,
                uncertainty=0.5,
                possible_abilities=("Synchronize", "Early Bird"),
                possible_items=("Leftovers",),
                possible_moves=("wish", "thunderwave", "psychic"),
            )
        return super().summarize(
            format_id=format_id,
            species=species,
            revealed_moves=revealed_moves,
            **kwargs,
        )


class ShowdownReplayNormalizationTest(unittest.TestCase):
    def test_timestamp_lines_are_not_normalized_into_public_events(self) -> None:
        replay = parse_showdown_replay(
            [
                "|player|p1|HumanFriend|",
                "|player|p2|PokeZeroBot|",
                "|t:|1782513831",
                "|switch|p1a: Xatu|Xatu, L80|100/100",
                "|t:|1782513832",
                "|turn|1",
            ],
            battle_id="battle-gen3randombattle-1",
        )

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        self.assertNotIn("|t:|", "\n".join(state.recent_public_events))
        self.assertFalse(any(event.event_type == "t:" for event in replay.public_events))
        self.assertFalse(any(line.startswith("|t:|") for line in replay.public_lines))

    def test_detected_player_name_overrides_stale_configured_slot(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        detected_slot = detect_showdown_slot(
            replay,
            player_name="PokeZeroBot",
            configured_showdown_slot="p1",
        )

        self.assertEqual(detected_slot, "p2")

    def test_p2_observation_is_player_relative_not_protocol_relative(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(
            replay,
            player_id="agent",
            player_name="PokeZeroBot",
            configured_showdown_slot="p1",
        )

        self.assertEqual(state.perspective.showdown_slot, "p2")
        self.assertEqual(state.perspective.opponent_showdown_slot, "p1")
        self.assertEqual(state.self_active.species, "Charizard")
        self.assertEqual(state.opponent_active.species, "Xatu")
        self.assertEqual(state.request_kind, "move")
        self.assertEqual(len(state.legal_action_mask), ACTION_COUNT)
        self.assertTrue(state.legal_action_mask[0])
        self.assertTrue(state.legal_action_mask[1])
        self.assertFalse(state.legal_action_mask[2])
        self.assertTrue(state.legal_action_mask[3])
        self.assertTrue(state.legal_action_mask[4])
        self.assertFalse(state.legal_action_mask[5])

    def test_two_players_receive_mirrored_self_and_opponent_views(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        p1_state = normalize_for_player(replay, player_id="human", player_name="HumanFriend")
        p2_state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        self.assertEqual(p1_state.self_active.species, "Xatu")
        self.assertEqual(p1_state.opponent_active.species, "Charizard")
        self.assertEqual(p2_state.self_active.species, "Charizard")
        self.assertEqual(p2_state.opponent_active.species, "Xatu")
        self.assertEqual(p1_state.perspective.showdown_slot, p2_state.perspective.opponent_showdown_slot)
        self.assertEqual(p2_state.perspective.showdown_slot, p1_state.perspective.opponent_showdown_slot)

    def test_hidden_opponent_request_state_is_not_exposed_in_opponent_team(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        opponent_species = {pokemon.species for pokemon in state.opponent_team}
        self.assertEqual(opponent_species, {"Arcanine", "Xatu"})
        self.assertNotIn("Alakazam", opponent_species)

    def test_previously_revealed_opponent_pokemon_remain_in_public_memory(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        self.assertEqual([pokemon.species for pokemon in state.opponent_team], ["Arcanine", "Xatu"])
        self.assertFalse(state.opponent_team[0].active)
        self.assertTrue(state.opponent_team[1].active)
        self.assertEqual(state.opponent_active.species, "Xatu")
        self.assertEqual(state.belief_view.opponent_pokemon[1].revealed_moves, ("Psychic",))
        self.assertEqual(state.belief_view.opponent_pokemon[1].condition, "70/100")

    def test_observation_shell_carries_detected_perspective_and_legal_mask(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)

        observation.validate(DEFAULT_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(observation.perspective.showdown_slot, "p2")
        self.assertEqual(observation.perspective.opponent_showdown_slot, "p1")
        self.assertEqual(observation.legal_action_mask, state.legal_action_mask)
        self.assertEqual(observation.metadata["self_active"]["species"], "Charizard")
        self.assertEqual(observation.metadata["opponent_active"]["species"], "Xatu")
        self.assertEqual(observation.metadata["action_candidates"][0]["move_name"], "flamethrower")
        self.assertEqual(observation.metadata["action_candidates"][4]["pokemon"]["species"], "Snorlax")

    def test_revealed_opponent_moves_populate_move_buckets_without_set_source(self) -> None:
        # Regression: revealed opponent moves are protocol ground truth and must be encoded even
        # when the belief set source is off (possible_moves empty). Previously the move buckets were
        # fed only possible_moves, so revealed moves never reached the persistent per-mon token —
        # the model saw the revealed-move COUNT but never which moves.
        from pokezero.showdown import CATEGORY_BELIEF_MOVE_OFFSET

        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")
        xatu = state.belief_view.opponent_pokemon[1]
        self.assertEqual(xatu.revealed_moves, ("Psychic",))
        self.assertEqual(xatu.possible_moves, ())  # no set source wired in this path

        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)
        opponent_offset = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT
        xatu_row = observation.categorical_ids[opponent_offset + 1]
        self.assertEqual(
            xatu_row[CATEGORY_BELIEF_MOVE_OFFSET],
            stable_category_id("belief:possible_move:psychic"),
        )

    @unittest.skipUnless(
        Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown/data/random-battles/gen3/sets.json").exists(),
        "requires a real Gen 3 Showdown checkout for the dex + vocab",
    )
    def test_transformed_ditto_encodes_target_stats_but_original_hp(self) -> None:
        # Ditto transforms into our Snorlax. Transform copies battle stats + types but NOT HP, so the
        # opponent Ditto token must show Snorlax's Attack yet keep Ditto's (frail) base HP.
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary
        from pokezero.showdown import NUMERIC_ACTIVE

        root = "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
        dex = load_showdown_dex_cached(root)
        vocab = gen3_category_vocabulary(root)
        lines = [
            "|player|p1|Us|",
            "|player|p2|Them|",
            "|switch|p1a: Snorlax|Snorlax, L78|100/100",
            "|switch|p2a: Ditto|Ditto, L78|100/100",
            "|turn|1",
            "|move|p2a: Ditto|Transform|p1a: Snorlax",
            "|-transform|p2a: Ditto|p1a: Snorlax",
            "|turn|2",
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(replay, player_id="agent", configured_showdown_slot="p1")
        obs = observation_from_player_state(state, category_vocab=vocab, dex=dex)

        opponent_offset = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT
        ditto = next(
            obs.numeric_features[opponent_offset + i]
            for i in range(OPPONENT_POKEMON_TOKEN_COUNT)
            if obs.numeric_features[opponent_offset + i][NUMERIC_ACTIVE] == 1.0
        )
        self.assertAlmostEqual(ditto[NUMERIC_BASE_ATK], 110 / 200)  # Snorlax's attack (copied)
        self.assertAlmostEqual(ditto[NUMERIC_BASE_HP], 48 / 200)  # Ditto's HP (NOT copied)

    @unittest.skipUnless(
        Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown/data/random-battles/gen3/sets.json").exists(),
        "requires a real Gen 3 Showdown checkout for the dex + vocab",
    )
    def test_ditto_transform_lifecycle_encoding_coverage(self) -> None:
        # Full Ditto lifecycle through the production encoder: while transformed it shows the
        # target's battle stats (Snorlax Attack) with its own HP; once it switches out it reverts
        # to Ditto's own stats. Guards both the transform masking and the switch-out reset.
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary
        from pokezero.showdown import CATEGORY_PRIMARY, NUMERIC_ACTIVE

        root = "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
        dex = load_showdown_dex_cached(root)
        vocab = gen3_category_vocabulary(root)
        base = [
            "|player|p1|Us|",
            "|player|p2|Them|",
            "|switch|p1a: Snorlax|Snorlax, L78|100/100",
            "|switch|p2a: Ditto|Ditto, L78|100/100",
            "|turn|1",
            "|move|p2a: Ditto|Transform|p1a: Snorlax",
            "|-transform|p2a: Ditto|p1a: Snorlax",
            "|turn|2",
        ]
        after_switch = base + [
            "|switch|p2a: Gengar|Gengar, L78|100/100",  # Ditto leaves -> reverts on the bench
            "|turn|3",
        ]
        opponent_offset = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT

        def opponent_tokens(lines):
            replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
            state = normalize_for_player(replay, player_id="agent", configured_showdown_slot="p1")
            obs = observation_from_player_state(state, category_vocab=vocab, dex=dex)
            return [obs.numeric_features[opponent_offset + i] for i in range(OPPONENT_POKEMON_TOKEN_COUNT)], \
                   [obs.categorical_ids[opponent_offset + i] for i in range(OPPONENT_POKEMON_TOKEN_COUNT)], obs

        # While transformed: the active opponent (Ditto) fights as Snorlax.
        num, _, _ = opponent_tokens(base)
        transformed_token = next(row for row in num if row[NUMERIC_ACTIVE] == 1.0)
        self.assertAlmostEqual(transformed_token[NUMERIC_BASE_ATK], 110 / 200)  # Snorlax
        self.assertAlmostEqual(transformed_token[NUMERIC_BASE_HP], 48 / 200)  # Ditto's HP

        # After switch-out: the benched Ditto has reverted to itself.
        num, cat, _ = opponent_tokens(after_switch)
        ditto_species = vocab.encode("species:Ditto")
        ditto_idx = next(i for i, row in enumerate(cat) if row[CATEGORY_PRIMARY] == ditto_species)
        self.assertEqual(num[ditto_idx][NUMERIC_ACTIVE], 0.0)
        self.assertAlmostEqual(num[ditto_idx][NUMERIC_BASE_ATK], 48 / 200)  # Ditto again, not Snorlax

    def test_side_conditions_are_player_relative_in_metadata(self) -> None:
        lines = [
            *fixture_lines("p2_seat_replay.txt")[:5],
            "|-sidestart|p1: HumanFriend|Spikes",
            "|-sidestart|p2: PokeZeroBot|Spikes",
            "|-sidestart|p2: PokeZeroBot|move: Stealth Rock",
            "|-sideend|p1: HumanFriend|Spikes",
            *fixture_lines("p2_seat_replay.txt")[5:],
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")
        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)

        self.assertEqual(state.self_side_conditions, ("spikes", "stealthrock"))
        self.assertEqual(state.opponent_side_conditions, ())
        self.assertEqual(observation.metadata["self_side_conditions"], ["spikes", "stealthrock"])
        self.assertEqual(observation.metadata["opponent_side_conditions"], [])

    def test_side_condition_layer_counts_are_player_relative_in_metadata(self) -> None:
        # Spikes is the only multi-layer side condition in Gen 3 and caps at 3 layers; a
        # 4th -sidestart must not push the count past 3. Reflect is single-layer.
        lines = [
            *fixture_lines("p2_seat_replay.txt")[:5],
            "|-sidestart|p1: HumanFriend|Spikes",
            "|-sidestart|p1: HumanFriend|Spikes",
            "|-sidestart|p1: HumanFriend|Spikes",
            "|-sidestart|p1: HumanFriend|Spikes",
            "|-sidestart|p2: PokeZeroBot|Spikes",
            "|-sidestart|p2: PokeZeroBot|Reflect",
            *fixture_lines("p2_seat_replay.txt")[5:],
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")
        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)

        self.assertEqual(state.self_side_conditions, ("reflect", "spikes"))
        self.assertEqual(state.opponent_side_conditions, ("spikes",))
        self.assertEqual(state.self_side_condition_counts, {"reflect": 1, "spikes": 1})
        self.assertEqual(state.opponent_side_condition_counts, {"spikes": 3})
        self.assertEqual(observation.metadata["self_side_condition_counts"], {"reflect": 1, "spikes": 1})
        self.assertEqual(observation.metadata["opponent_side_condition_counts"], {"spikes": 3})

    def test_observation_encodes_player_relative_content(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)
        self_offset = FIELD_TOKEN_COUNT
        opponent_offset = self_offset + SELF_POKEMON_TOKEN_COUNT
        action_offset = opponent_offset + OPPONENT_POKEMON_TOKEN_COUNT
        event_offset = action_offset + ACTION_CANDIDATE_TOKEN_COUNT

        self.assertEqual(observation.categorical_ids[0][0], stable_category_id("request_kind:move"))
        self.assertEqual(observation.categorical_ids[self_offset][0], stable_category_id("species:Charizard"))
        self.assertEqual(observation.numeric_features[self_offset][0], 1.0)
        self.assertEqual(observation.numeric_features[self_offset][1], 1.0)
        self.assertEqual(observation.categorical_ids[opponent_offset][0], stable_category_id("species:Arcanine"))
        self.assertEqual(observation.numeric_features[opponent_offset][1], 0.0)
        self.assertEqual(observation.categorical_ids[opponent_offset + 1][0], stable_category_id("species:Xatu"))
        self.assertEqual(observation.numeric_features[opponent_offset + 1][0], 0.7)
        self.assertEqual(observation.numeric_features[opponent_offset + 1][1], 1.0)
        self.assertEqual(observation.numeric_features[opponent_offset + 1][4], 1.0)
        self.assertEqual(observation.numeric_features[opponent_offset + 1][5], 0.0)
        self.assertEqual(observation.numeric_features[opponent_offset + 1][6], 1.0)
        self.assertEqual(observation.categorical_ids[action_offset][0], stable_category_id("move:flamethrower"))
        self.assertEqual(observation.numeric_features[action_offset][2], 1.0)
        self.assertEqual(observation.categorical_ids[action_offset + 2][0], stable_category_id("move:dragonclaw"))
        self.assertEqual(observation.numeric_features[action_offset + 2][1], 0.0)
        self.assertEqual(observation.numeric_features[action_offset + 2][2], 0.0)
        self.assertEqual(observation.categorical_ids[action_offset + 4][0], stable_category_id("species:Snorlax"))
        self.assertEqual(observation.numeric_features[action_offset + 4][2], 1.0)
        self.assertEqual(observation.categorical_ids[action_offset + 5][0], stable_category_id("species:Blissey"))
        self.assertEqual(observation.numeric_features[action_offset + 5][0], 0.0)
        self.assertEqual(observation.numeric_features[action_offset + 5][2], 0.0)
        self.assertEqual(observation.categorical_ids[event_offset][0], stable_category_id("event:player"))
        move_event_index = next(
            index for index, event in enumerate(state.recent_events) if event.event_type == "move"
        )
        damage_event_index = next(
            index for index, event in enumerate(state.recent_events) if event.event_type == "-damage"
        )
        self.assertEqual(
            observation.categorical_ids[event_offset + move_event_index][0],
            stable_category_id("event:move"),
        )
        self.assertEqual(
            observation.categorical_ids[event_offset + move_event_index][1],
            stable_category_id("move:Flamethrower"),
        )
        self.assertEqual(
            observation.categorical_ids[event_offset + move_event_index][2],
            stable_category_id("event_actor:self"),
        )
        self.assertEqual(
            observation.categorical_ids[event_offset + move_event_index][3],
            stable_category_id("event_target:opponent"),
        )
        self.assertEqual(
            observation.categorical_ids[event_offset + damage_event_index][0],
            stable_category_id("event:-damage"),
        )
        # Lean encoding: the -damage detail (raw HP string "70/100") is unactionable — HP is
        # captured numerically and status via -status events — so the SECONDARY slot is padding.
        self.assertEqual(observation.categorical_ids[event_offset + damage_event_index][1], 0)
        self.assertEqual(
            observation.categorical_ids[event_offset + damage_event_index][3],
            stable_category_id("event_target:opponent"),
        )

    def test_observation_encodes_public_belief_summary_features(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(
            replay,
            player_id="agent",
            player_name="PokeZeroBot",
            format_id="gen3randombattle",
            set_source=FakeSetSource(),
        )

        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)
        opponent_offset = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT
        xatu_offset = opponent_offset + 1

        self.assertEqual(observation.numeric_features[xatu_offset][5], 2.0)
        self.assertEqual(observation.numeric_features[xatu_offset][6], 0.5)
        self.assertEqual(observation.numeric_features[xatu_offset][7], 2.0)
        self.assertEqual(observation.numeric_features[xatu_offset][8], 1.0)
        self.assertEqual(observation.numeric_features[xatu_offset][9], 3.0)
        self.assertEqual(observation.numeric_features[xatu_offset][10], 0.0)
        self.assertEqual(observation.numeric_features[xatu_offset][11], 0.0)
        belief_fact_ids = observation.categorical_ids[xatu_offset][4:]
        self.assertIn(stable_category_id("belief:possible_ability:earlybird"), belief_fact_ids)
        self.assertIn(stable_category_id("belief:possible_ability:synchronize"), belief_fact_ids)
        self.assertIn(stable_category_id("belief:possible_item:leftovers"), belief_fact_ids)
        self.assertIn(stable_category_id("belief:possible_move:psychic"), belief_fact_ids)
        self.assertIn(stable_category_id("belief:possible_move:thunderwave"), belief_fact_ids)
        self.assertIn(stable_category_id("belief:possible_move:wish"), belief_fact_ids)
        self.assertNotIn(stable_category_id("belief:possible_moves:psychic|thunderwave|wish"), belief_fact_ids)
        self.assertNotIn("belief", observation.metadata)
        self.assertNotIn("belief", observation.metadata["opponent_active"])

        reordered_state = normalize_for_player(
            replay,
            player_id="agent",
            player_name="PokeZeroBot",
            format_id="gen3randombattle",
            set_source=ReorderedFakeSetSource(),
        )
        reordered = observation_from_player_state(reordered_state, category_vocab=_TEST_VOCAB)
        self.assertEqual(
            observation.categorical_ids[xatu_offset],
            reordered.categorical_ids[xatu_offset],
        )

    def test_policy_action_translates_back_to_showdown_choice_for_detected_side(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        self.assertEqual(showdown_choice_for_action(state, 1), "move 2")
        self.assertEqual(showdown_choice_for_action(state, 4), "switch 2")
        self.assertEqual(showdown_submission_for_action(state, 4).showdown_slot, "p2")
        self.assertEqual(showdown_submission_for_action(state, 4).choice, "switch 2")

        with self.assertRaisesRegex(ValueError, "not legal"):
            showdown_choice_for_action(state, 2)

    def test_recent_events_are_normalized_to_self_and_opponent_roles(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        joined_events = "\n".join(state.recent_public_events)
        self.assertIn("|player|p1|HumanFriend|", joined_events)
        self.assertIn("|player|p2|PokeZeroBot|", joined_events)
        self.assertIn("|move|selfa: Charizard|Flamethrower|opponenta: Xatu", joined_events)
        self.assertIn("|-damage|opponenta: Xatu|70/100", joined_events)
        self.assertIn("opponenta: Xatu", joined_events)
        self.assertIn("selfa: Charizard", joined_events)
        self.assertNotIn("|player|opponent|", joined_events)
        self.assertNotIn("|player|self|", joined_events)
        self.assertNotIn("p1a: Xatu", joined_events)
        self.assertNotIn("p2a: Charizard", joined_events)

    def test_recent_events_are_structured_before_rendering(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")
        move_event = next(event for event in state.recent_events if event.event_type == "move")
        damage_event = next(event for event in state.recent_events if event.event_type == "-damage")

        self.assertEqual(move_event.actor_role, "self")
        self.assertEqual(move_event.target_role, "opponent")
        self.assertEqual(move_event.primary, "Flamethrower")
        self.assertEqual(damage_event.actor_role, "none")
        self.assertEqual(damage_event.target_role, "opponent")
        self.assertEqual(damage_event.primary, "70/100")


class Phase2DynamicStateTest(unittest.TestCase):
    """Phase 2 dynamic decision-critical state: level, base stats, boosts, weather, hazards."""

    SELF_ACTIVE_OFFSET = FIELD_TOKEN_COUNT  # token 1: first self mon (active Charizard).
    ACTION_OFFSET = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT + OPPONENT_POKEMON_TOKEN_COUNT

    def _replay_with(self, extra_lines: list[str]):
        lines = [
            *fixture_lines("p2_seat_replay.txt")[:5],
            *extra_lines,
            *fixture_lines("p2_seat_replay.txt")[5:],
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
        return normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

    def test_level_and_base_stats_on_active_self_mon(self) -> None:
        state = self._replay_with([])
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, dex=_phase2_fake_dex()
        )
        numeric = observation.numeric_features[self.SELF_ACTIVE_OFFSET]
        self.assertAlmostEqual(numeric[NUMERIC_LEVEL], 0.78)
        self.assertAlmostEqual(numeric[NUMERIC_BASE_HP], 78 / 200)
        self.assertAlmostEqual(numeric[NUMERIC_BASE_ATK], 84 / 200)
        self.assertAlmostEqual(numeric[NUMERIC_BASE_DEF], 78 / 200)
        self.assertAlmostEqual(numeric[NUMERIC_BASE_SPA], 109 / 200)
        self.assertAlmostEqual(numeric[NUMERIC_BASE_SPD], 85 / 200)
        self.assertAlmostEqual(numeric[NUMERIC_BASE_SPE], 100 / 200)

    def test_level_present_without_dex_but_base_stats_padding(self) -> None:
        state = self._replay_with([])
        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)
        numeric = observation.numeric_features[self.SELF_ACTIVE_OFFSET]
        self.assertAlmostEqual(numeric[NUMERIC_LEVEL], 0.78)  # parsed from details, no dex needed
        self.assertEqual(numeric[NUMERIC_BASE_SPA], 0.0)  # base stats need the dex

    def test_weather_parsed_and_encoded_on_field_token(self) -> None:
        state = self._replay_with(["|-weather|RainDance"])
        self.assertEqual(state.weather, "raindance")
        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)
        self.assertEqual(
            observation.categorical_ids[FIELD_TOKEN_OFFSET][CATEGORY_SECONDARY],
            stable_category_id("weather:raindance"),
        )

    def test_weather_none_clears(self) -> None:
        state = self._replay_with(["|-weather|RainDance", "|-weather|none"])
        self.assertIsNone(state.weather)

    def test_boosts_accumulate_on_active_mon(self) -> None:
        state = self._replay_with(
            [
                "|-boost|p2a: Charizard|atk|2",
                "|-boost|p2a: Charizard|atk|1",
                "|-unboost|p2a: Charizard|spe|1",
            ]
        )
        self.assertEqual(state.self_active_boosts, {"atk": 3, "spe": -1})
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, dex=_phase2_fake_dex()
        )
        numeric = observation.numeric_features[self.SELF_ACTIVE_OFFSET]
        self.assertAlmostEqual(numeric[NUMERIC_BOOST_ATK], 3 / 6)
        self.assertAlmostEqual(numeric[NUMERIC_BOOST_SPE], -1 / 6)

    def test_boosts_clamp_and_reset_on_switch(self) -> None:
        # +8 worth of boosts clamps to +6; a later switch-in wipes the slot back to zero.
        state = self._replay_with(
            [
                "|-boost|p2a: Charizard|atk|6",
                "|-boost|p2a: Charizard|atk|2",
            ]
        )
        self.assertEqual(state.self_active_boosts, {"atk": 6})

        reset_state = self._replay_with(
            [
                "|-boost|p2a: Charizard|atk|6",
                "|switch|p2a: Snorlax|Snorlax, L78|100/100",
                "|switch|p2a: Charizard|Charizard, L78|100/100",
            ]
        )
        self.assertEqual(reset_state.self_active_boosts, {})

    def test_baton_pass_carries_boosts_to_incoming_mon(self) -> None:
        # Charizard sets +2 Atk then Baton Passes to Snorlax: Snorlax inherits the boost.
        state = self._replay_with(
            [
                "|-boost|p2a: Charizard|atk|2",
                "|move|p2a: Charizard|Baton Pass",
                "|switch|p2a: Snorlax|Snorlax, L78|100/100",
            ]
        )
        self.assertEqual(state.self_active_boosts, {"atk": 2})

    def test_baton_pass_via_switch_from_tag_carries_boosts(self) -> None:
        # Some replays only tag the switch line ("[from] Baton Pass") without a flag-setting
        # move line in the recent window; that tag alone must still carry boosts.
        state = self._replay_with(
            [
                "|-boost|p2a: Charizard|spe|2",
                "|switch|p2a: Snorlax|Snorlax, L78|100/100|[from] Baton Pass",
            ]
        )
        self.assertEqual(state.self_active_boosts, {"spe": 2})

    def test_psych_up_copies_opponent_boosts(self) -> None:
        # -copyboost: the self mon (p2) copies the opponent's (p1) boost stages.
        state = self._replay_with(
            [
                "|-boost|p1a: Xatu|spa|2",
                "|-boost|p1a: Xatu|spd|1",
                "|-copyboost|p2a: Charizard|p1a: Xatu|[from] move: Psych Up",
            ]
        )
        self.assertEqual(state.self_active_boosts, {"spa": 2, "spd": 1})
        self.assertEqual(state.opponent_active_boosts, {"spa": 2, "spd": 1})

    def test_normal_switch_after_unrelated_move_resets_boosts(self) -> None:
        # A non-Baton-Pass move before the switch must NOT carry boosts.
        state = self._replay_with(
            [
                "|-boost|p2a: Charizard|atk|2",
                "|move|p2a: Charizard|Earthquake|p1a: Xatu",
                "|switch|p2a: Snorlax|Snorlax, L78|100/100",
            ]
        )
        self.assertEqual(state.self_active_boosts, {})

    def test_setboost_overwrites_stage(self) -> None:
        state = self._replay_with(
            [
                "|-boost|p2a: Charizard|atk|2",
                "|-setboost|p2a: Charizard|atk|6",  # Belly Drum-style absolute set
            ]
        )
        self.assertEqual(state.self_active_boosts, {"atk": 6})

    def test_move_effect_type_chance_and_hp_cost(self) -> None:
        state = self._replay_with([])
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, dex=_phase2_fake_dex()
        )
        move_token = self.ACTION_OFFSET  # first move action token (Flamethrower).
        self.assertEqual(
            observation.categorical_ids[move_token][CATEGORY_MOVE_EFFECT],
            stable_category_id("move_effect:brn"),
        )
        self.assertAlmostEqual(observation.numeric_features[move_token][NUMERIC_EFFECT_CHANCE], 0.10)
        self.assertAlmostEqual(observation.numeric_features[move_token][NUMERIC_SELF_HP_COST], 0.0)

    def test_curse_move_effect_resolves_by_user_type(self) -> None:
        # The acting mon's type decides Curse's encoded effect/cost (stable within a battle).
        curse = MoveInfo(
            id="curse", name="Curse", type="Ghost", category="Status", gen3_category="Status",
            base_power=0, accuracy=100.0, priority=0, recoil=False, drain=False, heal=False,
            status=None, boosts={}, target="normal", selfdestruct=False,
            effect_chance=0, effect_label="", self_hp_cost=0.0,  # static label suppressed
        )
        dex = ShowdownDex(moves={"curse": curse}, species={}, type_chart={})
        spec = DEFAULT_REPLAY_OBSERVATION_SPEC

        def _encode(user_types):
            cat = [""] * spec.categorical_feature_count
            num = [0.0] * spec.numeric_feature_count
            _encode_move_mechanics(cat, num, dex, "curse", user_types)
            return cat[CATEGORY_MOVE_EFFECT], num[NUMERIC_SELF_HP_COST]

        self.assertEqual(_encode(("Ghost",)), ("move_effect:curse", 0.5))
        self.assertEqual(_encode(("Normal",)), ("move_effect:curse_setup", 0.0))

    def test_move_pp_fraction_helper(self) -> None:
        self.assertAlmostEqual(_move_pp_fraction({"pp": 8, "maxpp": 8}), 1.0)
        self.assertAlmostEqual(_move_pp_fraction({"pp": 1, "maxpp": 8}), 0.125)
        self.assertAlmostEqual(_move_pp_fraction({"pp": 0, "maxpp": 8}), 0.0)
        self.assertAlmostEqual(_move_pp_fraction({}), 1.0)  # absent PP data -> assume full
        self.assertAlmostEqual(_move_pp_fraction({"pp": 5, "maxpp": 0}), 1.0)  # guard div-by-zero

    def test_move_pp_fraction_defaults_full_without_request_pp(self) -> None:
        # The fixture request omits pp/maxpp, so the encoded fraction defaults to full (1.0).
        state = self._replay_with([])
        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)
        self.assertAlmostEqual(
            observation.numeric_features[self.ACTION_OFFSET][NUMERIC_MOVE_PP_FRACTION], 1.0
        )

    def test_future_sight_tracked_as_incoming_and_outgoing(self) -> None:
        # Bot is p2; p2's Future Sight lands on p1 (opponent) — the player's OUTGOING delayed hit.
        state = self._replay_with(
            [
                "|turn|5",
                "|move|p2a: Charizard|Future Sight|p1a: Xatu",
                "|-start|p2a: Charizard|move: Future Sight",
            ]
        )
        self.assertEqual(state.opponent_future_sight_turns, 2)
        self.assertEqual(state.self_future_sight_turns, 0)
        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)
        numeric = observation.numeric_features[FIELD_TOKEN_OFFSET]
        self.assertAlmostEqual(numeric[NUMERIC_OPP_FUTURE_SIGHT], 1.0)  # 2 turns / 2
        self.assertAlmostEqual(numeric[NUMERIC_SELF_FUTURE_SIGHT], 0.0)

    def test_future_sight_survives_switch(self) -> None:
        # Future Sight is a side-level slot condition: the user switching out must NOT clear it.
        state = self._replay_with(
            [
                "|turn|5",
                "|move|p2a: Charizard|Future Sight|p1a: Xatu",
                "|-start|p2a: Charizard|move: Future Sight",
                "|switch|p2a: Snorlax|Snorlax, L78|100/100",
            ]
        )
        self.assertEqual(state.opponent_future_sight_turns, 2)

    def test_toxic_stage_escalates_resets_and_ignores_regular_poison(self) -> None:
        # tox escalates each turn (1 on apply, +1 per turn); both sides are tracked the same way.
        state = self._replay_with(["|-status|p2a: Charizard|tox", "|turn|6", "|turn|7"])
        self.assertEqual(state.self_toxic_stage, 3)
        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)
        self.assertAlmostEqual(
            observation.numeric_features[self.SELF_ACTIVE_OFFSET][NUMERIC_TOXIC_STAGE], 3 / 15
        )
        # Regular poison does not escalate.
        psn = self._replay_with(["|-status|p2a: Charizard|psn", "|turn|6", "|turn|7"])
        self.assertEqual(psn.self_toxic_stage, 0)
        # Switching out resets the toxic counter (Gen 3).
        reset = self._replay_with(
            ["|-status|p2a: Charizard|tox", "|turn|6", "|switch|p2a: Snorlax|Snorlax, L78|100/100"]
        )
        self.assertEqual(reset.self_toxic_stage, 0)

    def test_future_sight_cleared_when_it_lands(self) -> None:
        landed = self._replay_with(
            [
                "|turn|5",
                "|move|p2a: Charizard|Future Sight|p1a: Xatu",
                "|-start|p2a: Charizard|move: Future Sight",
                "|turn|7",
                "|-end|p1a: Xatu|move: Future Sight",
            ]
        )
        self.assertEqual(landed.opponent_future_sight_turns, 0)

    def test_turn_count_on_field_token(self) -> None:
        state = self._replay_with(["|turn|7"])
        self.assertEqual(state.turn_number, 7)
        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)
        self.assertAlmostEqual(
            observation.numeric_features[FIELD_TOKEN_OFFSET][NUMERIC_TURN_COUNT], 7 / 1000
        )

    def test_volatiles_tracked_and_encoded_on_active_mon(self) -> None:
        state = self._replay_with(
            [
                "|-start|p2a: Charizard|confusion",
                "|-start|p2a: Charizard|move: Leech Seed",
            ]
        )
        self.assertEqual(state.self_active_volatiles, ("confusion", "leechseed"))
        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)
        volatile_cols = observation.categorical_ids[self.SELF_ACTIVE_OFFSET][
            CATEGORY_VOLATILE_OFFSET : CATEGORY_VOLATILE_OFFSET + 6
        ]
        self.assertIn(stable_category_id("volatile:confusion"), volatile_cols)
        self.assertIn(stable_category_id("volatile:leechseed"), volatile_cols)

    def test_volatile_strips_ability_prefix_and_filters_non_volatiles(self) -> None:
        # "ability: Flash Fire" must normalize to the bare tracked id (not "abilityflashfire"),
        # and an untracked -start payload (typechange) must be ignored, not encoded as a volatile.
        state = self._replay_with(
            [
                "|-start|p2a: Charizard|ability: Flash Fire",
                "|-start|p2a: Charizard|typechange|Fire",
            ]
        )
        self.assertEqual(state.self_active_volatiles, ("flashfire",))

    def test_volatiles_cleared_on_end_and_switch(self) -> None:
        ended = self._replay_with(
            [
                "|-start|p2a: Charizard|confusion",
                "|-end|p2a: Charizard|confusion",
            ]
        )
        self.assertEqual(ended.self_active_volatiles, ())

        switched = self._replay_with(
            [
                "|-start|p2a: Charizard|confusion",
                "|switch|p2a: Snorlax|Snorlax, L78|100/100",
            ]
        )
        self.assertEqual(switched.self_active_volatiles, ())

    def test_hazards_and_screens_on_field_token(self) -> None:
        # Bot is p2 (self). Opponent (p1) sets 3 Spikes; self sets Reflect + Light Screen.
        state = self._replay_with(
            [
                "|-sidestart|p1: HumanFriend|Spikes",
                "|-sidestart|p1: HumanFriend|Spikes",
                "|-sidestart|p1: HumanFriend|Spikes",
                "|-sidestart|p2: PokeZeroBot|Reflect",
                "|-sidestart|p2: PokeZeroBot|Light Screen",
            ]
        )
        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)
        numeric = observation.numeric_features[FIELD_TOKEN_OFFSET]
        self.assertAlmostEqual(numeric[NUMERIC_OPP_HAZARDS], 1.0)  # 3 spikes / 3
        self.assertAlmostEqual(numeric[NUMERIC_SELF_HAZARDS], 0.0)
        self.assertAlmostEqual(numeric[NUMERIC_SELF_SCREENS], 1.0)  # reflect + light screen / 2
        self.assertAlmostEqual(numeric[NUMERIC_OPP_SCREENS], 0.0)


class PlayerActualStatsTest(unittest.TestCase):
    """The player's own actual computed stats (from the request) are surfaced on self tokens."""

    def test_max_hp_from_condition(self) -> None:
        self.assertEqual(_max_hp_from_condition("180/250"), 250)
        self.assertEqual(_max_hp_from_condition("250/250"), 250)
        self.assertIsNone(_max_hp_from_condition("0 fnt"))
        self.assertIsNone(_max_hp_from_condition(None))

    def test_actual_stats_from_request_row(self) -> None:
        row = {"condition": "250/250", "stats": {"atk": 200, "def": 180, "spa": 286, "spd": 206, "spe": 236}}
        self.assertEqual(
            _actual_stats_from_request_row(row, row["condition"]),
            {"atk": 200, "def": 180, "spa": 286, "spd": 206, "spe": 236, "hp": 250},
        )
        # No stats object (e.g. simplified payload) -> None.
        self.assertIsNone(_actual_stats_from_request_row({"condition": "0 fnt"}, "0 fnt"))

    def test_actual_stats_encoded_on_self_tokens_only(self) -> None:
        request = {
            "active": [{"moves": [{"move": "Flamethrower", "id": "flamethrower", "pp": 8, "maxpp": 8}]}],
            "side": {
                "id": "p2",
                "name": "PokeZeroBot",
                "pokemon": [
                    {"ident": "p2a: Charizard", "details": "Charizard, L78", "condition": "250/250",
                     "active": True, "stats": {"atk": 200, "def": 180, "spa": 286, "spd": 206, "spe": 236}},
                    {"ident": "p2b: Snorlax", "details": "Snorlax, L78", "condition": "520/520",
                     "active": False, "stats": {"atk": 250, "def": 160, "spa": 160, "spd": 230, "spe": 90}},
                ],
            },
        }
        lines = [
            "|player|p1|Foe|1|",
            "|player|p2|PokeZeroBot|2|",
            "|switch|p1a: Xatu|Xatu, L78|100/100",
            "|switch|p2a: Charizard|Charizard, L78|250/250",
            "|turn|1",
            "|request|" + json.dumps(request),
        ]
        replay = parse_showdown_replay(lines, battle_id="b")
        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")
        observation = observation_from_player_state(state, category_vocab=_TEST_VOCAB)

        self_active = FIELD_TOKEN_COUNT  # token 1: active Charizard
        self_bench = FIELD_TOKEN_COUNT + 1  # token 2: Snorlax
        opp_active = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT  # token 7: Xatu (no actual stats)
        self.assertAlmostEqual(observation.numeric_features[self_active][NUMERIC_ACTUAL_SPE], 236 / 714)
        self.assertAlmostEqual(observation.numeric_features[self_active][NUMERIC_ACTUAL_HP], 250 / 714)
        self.assertAlmostEqual(observation.numeric_features[self_bench][NUMERIC_ACTUAL_SPE], 90 / 714)
        # The opponent's actual stats are hidden -> the slots stay padding (0).
        self.assertEqual(observation.numeric_features[opp_active][NUMERIC_ACTUAL_SPE], 0.0)
        self.assertEqual(observation.numeric_features[opp_active][NUMERIC_ACTUAL_HP], 0.0)


if __name__ == "__main__":
    unittest.main()
