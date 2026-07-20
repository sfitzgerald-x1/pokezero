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
    STATS_TOKEN_COUNT,
    ObservationFeatureMasks,
)
from pokezero.category_vocab import build_category_vocabulary
from pokezero.dex import MoveInfo, ShowdownDex, SpeciesInfo
from pokezero.showdown import (
    CATEGORY_MOVE_EFFECT,
    NUMERIC_TT_DAMAGE_FRACTION,
    CATEGORY_SECONDARY,
    CATEGORY_VOLATILE_OFFSET,
    DEFAULT_REPLAY_OBSERVATION_SPEC,
    V2_1_REPLAY_OBSERVATION_SPEC,
    NUMERIC_ACTUAL_HP,
    NUMERIC_ACTUAL_SPE,
    NUMERIC_EFFECT_CHANCE,
    NUMERIC_MOVE_PP_FRACTION,
    NUMERIC_OPP_FUTURE_SIGHT,
    NUMERIC_SELF_FUTURE_SIGHT,
    NUMERIC_SELF_HP_COST,
    NUMERIC_TOXIC_STAGE,
    NUMERIC_TURN_COUNT,
    NUMERIC_WEATHER_PERMANENT,
    NUMERIC_WEATHER_TURNS,
    CATEGORY_PRIMARY,
    CATEGORY_TYPE_1,
    CATEGORY_MOVE_CATEGORY,
    NUMERIC_BASE_POWER,
    _ReplayParser,
    _weather_duration_features,
    _actual_stats_from_request_row,
    _encode_move_mechanics,
    _hidden_power_variant_from_name,
    _self_move_mechanics_id,
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
    _pokemon_metadata,
    _self_team_from_request,
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
        "stats", "transition:self", "transition:opponent",
        "tt_kind:move", "tt_kind:switch", "tt_kind:cant",
        "tt_outcome:normal", "tt_effectiveness:neutral", "tt_side_effect:none",
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
    def test_public_damage_updates_the_active_pokemon_condition(self) -> None:
        replay = parse_showdown_replay(
            [
                "|switch|p1a: Charizard|Charizard, L80|250/250",
                "|switch|p2a: Xatu|Xatu, L80|220/220",
                "|-damage|p2a: Xatu|180/220 brn",
                "|-heal|p1a: Charizard|250/250",
                "|-status|p1a: Charizard|par",
            ]
        )

        self.assertEqual(replay.public_active["p2"].condition, "180/220 brn")
        self.assertEqual(replay.public_active["p1"].condition, "250/250 par")
        self.assertEqual(replay.public_revealed["p2"][0].condition, "180/220 brn")

    def test_cureteam_strips_status_suffix_team_wide_and_resets_toxic_stage(self) -> None:
        # Aromatherapy emits a single |-cureteam|SOURCE (no per-mon -curestatus). The public
        # condition update must strip the status suffix from the source's active mon AND every
        # revealed benched mon on that side, and reset that side's toxic ramp.
        replay = parse_showdown_replay(
            [
                "|switch|p1a: Swampert|Swampert, L84|300/300",
                "|switch|p2a: Vigoroth|Vigoroth, L84|301/301",
                "|-status|p2a: Vigoroth|tox",
                "|-damage|p2a: Vigoroth|283/301 tox|[from] psn",
                "|switch|p2a: Blissey|Blissey, L82|300/300",
                "|-status|p2a: Blissey|brn",
                "|-damage|p2a: Blissey|280/300 brn",
                "|move|p2a: Blissey|Aromatherapy|p2a: Blissey",
                "|-cureteam|p2a: Blissey|[from] move: Aromatherapy",
            ]
        )
        # Active source (Blissey): burn suffix stripped.
        self.assertEqual(replay.public_active["p2"].condition, "280/300")
        # Benched revealed Vigoroth: tox suffix stripped.
        vigoroth = next(p for p in replay.public_revealed["p2"] if p.species == "Vigoroth")
        self.assertEqual(vigoroth.condition, "283/301")
        # Toxic ramp on the cured side reset; the opponent side untouched.
        self.assertEqual(replay.toxic_stage["p2"], 0)

    def test_benched_heal_bell_curestatus_strips_off_field_ally_and_leaves_others(self) -> None:
        # Heal Bell (unlike Aromatherapy's team-wide -cureteam) emits a per-mon [silent]
        # -curestatus for EACH cured member: the active user carries a field-position letter
        # (p2a: Miltank) and every benched ally is POSITION-LESS (p2: Aggron). The active-only
        # public-condition path resolves public_active[slot] (the active Miltank), so without the
        # benched branch the off-field ally's status suffix in public_revealed never clears — the
        # parser-surface sibling of the belief engine's benched -curestatus handling (#771). Guard
        # both directions: the benched ally clears, but the ACTIVE-target cure, healthy same-side
        # members, and the OPPONENT side stay exactly as before.
        replay = parse_showdown_replay(
            [
                "|switch|p1a: Swampert|Swampert, L84|300/300",
                "|switch|p2a: Aggron|Aggron, L80|280/280",
                "|turn|1",
                "|move|p1a: Swampert|Toxic|p2a: Aggron",
                "|-status|p2a: Aggron|tox",
                "|-damage|p2a: Aggron|262/280 tox|[from] psn",
                "|move|p2a: Aggron|Toxic|p1a: Swampert",
                "|-status|p1a: Swampert|tox",
                "|-damage|p1a: Swampert|283/300 tox|[from] psn",
                "|turn|2",
                "|switch|p2a: Snorlax|Snorlax, L82|400/400",
                "|turn|3",
                "|switch|p2a: Miltank|Miltank, L82|300/300",
                "|-status|p2a: Miltank|par",
                "|turn|4",
                "|move|p2a: Miltank|Heal Bell|p2a: Miltank",
                "|-activate|p2a: Miltank|move: Heal Bell",
                "|-curestatus|p2a: Miltank|par|[silent]",
                "|-curestatus|p2: Aggron|tox|[silent]",
            ]
        )
        # Benched ally (Aggron): tox suffix stripped — the fix (fails on origin/main: "262/280 tox").
        aggron = next(p for p in replay.public_revealed["p2"] if p.species == "Aggron")
        self.assertEqual(aggron.condition, "262/280")
        # ACTIVE-target cure (Miltank, position-bearing ident): par stripped on the unchanged path.
        self.assertEqual(replay.public_active["p2"].condition, "300/300")
        # Healthy same-side member (Snorlax): no phantom status, untouched.
        snorlax = next(p for p in replay.public_revealed["p2"] if p.species == "Snorlax")
        self.assertEqual(snorlax.condition, "400/400")
        # NO OVER-CLEARING: the OPPONENT side's genuinely-toxic mon keeps its status suffix.
        self.assertEqual(replay.public_active["p1"].condition, "283/300 tox")
        # Cured side's toxic ramp reset; the opponent side's ramp is untouched.
        self.assertEqual(replay.toxic_stage["p2"], 0)
        self.assertEqual(replay.toxic_stage["p1"], 4)

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

        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )

        observation.validate(V2_1_REPLAY_OBSERVATION_SPEC)
        self.assertEqual(observation.perspective.showdown_slot, "p2")
        self.assertEqual(observation.perspective.opponent_showdown_slot, "p1")
        self.assertEqual(observation.legal_action_mask, state.legal_action_mask)
        self.assertEqual(observation.metadata["self_active"]["species"], "Charizard")
        self.assertEqual(observation.metadata["opponent_active"]["species"], "Xatu")
        self.assertEqual(observation.metadata["action_candidates"][0]["move_name"], "flamethrower")
        self.assertEqual(observation.metadata["action_candidates"][4]["pokemon"]["species"], "Snorlax")

    def test_self_team_request_metadata_carries_private_set_fields_for_search_materialization(self) -> None:
        request = {
            "active": [
                {
                    "moves": [
                        {"id": "fireblast", "move": "Fire Blast"},
                        {"id": "dragonclaw", "move": "Dragon Claw"},
                    ]
                }
            ],
            "side": {
                "id": "p2",
                "pokemon": [
                    {
                        "ident": "p2a: Charizard",
                        "details": "Charizard, L79",
                        "condition": "250/250",
                        "active": True,
                        "baseAbility": "Blaze",
                        "item": "Petaya Berry",
                        "stats": {"atk": 180, "def": 160, "spa": 240, "spd": 190, "spe": 220},
                    },
                    {
                        "ident": "p2b: Blissey",
                        "details": "Blissey, L75",
                        "condition": "300/300",
                        "moves": ["seismictoss", "softboiled", "toxic", "thunderwave"],
                        "ability": "Natural Cure",
                        "item": "Leftovers",
                    },
                ],
            },
        }

        team = _self_team_from_request(request, "p2")
        metadata = _pokemon_metadata(team[0])

        self.assertEqual(team[0].moves, ("fireblast", "dragonclaw"))
        self.assertEqual(team[0].ability, "Blaze")
        self.assertEqual(team[0].item, "Petaya Berry")
        self.assertEqual(team[1].moves, ("seismictoss", "softboiled", "toxic", "thunderwave"))
        self.assertEqual(metadata["moves"], ["fireblast", "dragonclaw"])
        self.assertEqual(metadata["ability"], "Blaze")
        self.assertEqual(metadata["item"], "Petaya Berry")
        self.assertEqual(metadata["stats"]["hp"], 250)

    def test_opponent_rows_carry_belief_revealed_facts(self) -> None:
        # Regression: opponent rows' moves/ability/item stayed permanently empty because reveals
        # accumulate only in the belief engine — metadata consumers (dataset shaping, probes)
        # silently saw none of what the encoder's belief facts saw. normalize_for_player now
        # merges revealed facts onto the public rows, normalized to identifier form to match
        # request-sourced self rows.
        lines = fixture_lines("p2_seat_replay.txt") + [
            "|-heal|p1a: Xatu|80/100|[from] item: Leftovers",
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")

        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        xatu = state.opponent_team[1]
        self.assertEqual(xatu.species, "Xatu")
        self.assertEqual(xatu.moves, ("psychic",))
        self.assertEqual(xatu.item, "leftovers")
        metadata = _pokemon_metadata(xatu)
        self.assertEqual(metadata["moves"], ["psychic"])
        self.assertEqual(metadata["item"], "leftovers")

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

        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        opponent_offset = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT
        xatu_row = observation.categorical_ids[opponent_offset + 1]
        self.assertEqual(
            xatu_row[CATEGORY_BELIEF_MOVE_OFFSET],
            stable_category_id("belief:possible_move:psychic"),
        )

    def test_removed_or_consumed_item_encodes_as_not_currently_held(self) -> None:
        # Regression (training-data-corruption guard): NUMERIC_REVEALED_ITEM is a CURRENT-held
        # signal, not a "was ever revealed" flag. A Knock Off / consumed berry leaves belief with
        # revealed_item still NAMING the item (it pins the opponent's set) but item_removed=True —
        # the mon holds nothing now. The encoder previously read `revealed_item` alone, so a
        # removed/eaten item stayed encoded as still-held (=1.0), corrupting self-play training at
        # high incidence (Leftovers is the default Gen 3 item; Knock Off is common; all pinch/status
        # berries). The current-held column must go to 0 on removal/consumption WHILE the possible_item
        # set-identity columns keep the reveal (so set inference survives the item leaving the field).
        from pokezero.showdown import (
            CATEGORY_BELIEF_ITEM_OFFSET,
            CATEGORY_PRIMARY,
            NUMERIC_POSSIBLE_ITEM_COUNT,
            NUMERIC_REVEALED_ITEM,
        )

        vocab = build_category_vocabulary(
            [
                "request_kind:move", "stats", "transition:self", "transition:opponent",
                "species:Charizard", "species:Snorlax",
                "belief:possible_item:leftovers", "belief:possible_item:salacberry",
            ]
        )
        opponent_offset = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT
        snorlax_species = vocab.encode("species:Snorlax")

        def snorlax_token(lines):
            replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
            state = normalize_for_player(replay, player_id="agent", configured_showdown_slot="p1")
            obs = observation_from_player_state(
                state, category_vocab=vocab, spec=V2_1_REPLAY_OBSERVATION_SPEC
            )
            idx = next(
                opponent_offset + i
                for i in range(OPPONENT_POKEMON_TOKEN_COUNT)
                if obs.categorical_ids[opponent_offset + i][CATEGORY_PRIMARY] == snorlax_species
            )
            return obs.numeric_features[idx], obs.categorical_ids[idx]

        base = [
            "|player|p1|Us|",
            "|player|p2|Them|",
            "|switch|p1a: Charizard|Charizard, L78|100/100",
            "|switch|p2a: Snorlax|Snorlax, L78|100/100",
            "|turn|1",
            "|-damage|p2a: Snorlax|90/100",
            "|-heal|p2a: Snorlax|100/100|[from] item: Leftovers",  # Leftovers revealed, HELD
            "|turn|2",
        ]
        # Sanity: while the item is genuinely held, current-held is 1.0.
        held_num, held_cat = snorlax_token(base)
        self.assertEqual(held_num[NUMERIC_REVEALED_ITEM], 1.0)
        self.assertEqual(held_cat[CATEGORY_BELIEF_ITEM_OFFSET], vocab.encode("belief:possible_item:leftovers"))
        self.assertEqual(held_num[NUMERIC_POSSIBLE_ITEM_COUNT], 1.0)

        # Knock Off: the item is publicly gone (item_removed). Current-held -> 0.0, but the
        # possible_item set-identity column and count MUST still name Leftovers (set inference kept).
        knocked = base + [
            "|move|p1a: Charizard|Knock Off|p2a: Snorlax",
            "|-enditem|p2a: Snorlax|Leftovers|[from] move: Knock Off|[of] p1a: Charizard",
            "|turn|3",
        ]
        ko_num, ko_cat = snorlax_token(knocked)
        self.assertEqual(ko_num[NUMERIC_REVEALED_ITEM], 0.0)  # <-- the guard: NOT still-held
        self.assertEqual(ko_cat[CATEGORY_BELIEF_ITEM_OFFSET], vocab.encode("belief:possible_item:leftovers"))
        self.assertEqual(ko_num[NUMERIC_POSSIBLE_ITEM_COUNT], 1.0)  # set inference preserved

        # Consumed berry (-enditem [eat], no [from] move): same current-held semantics.
        ate = [
            "|player|p1|Us|",
            "|player|p2|Them|",
            "|switch|p1a: Charizard|Charizard, L78|100/100",
            "|switch|p2a: Snorlax|Snorlax, L78|100/100",
            "|turn|1",
            "|-damage|p2a: Snorlax|20/100",
            "|-enditem|p2a: Snorlax|Salac Berry|[eat]",
            "|-boost|p2a: Snorlax|spe|1|[from] item: Salac Berry",
            "|turn|2",
        ]
        eat_num, eat_cat = snorlax_token(ate)
        self.assertEqual(eat_num[NUMERIC_REVEALED_ITEM], 0.0)  # consumed berry is not held
        self.assertEqual(eat_cat[CATEGORY_BELIEF_ITEM_OFFSET], vocab.encode("belief:possible_item:salacberry"))
        self.assertEqual(eat_num[NUMERIC_POSSIBLE_ITEM_COUNT], 1.0)  # eaten item still pins the set

    def test_self_known_item_and_ability_reach_the_self_observation_token(self) -> None:
        # Regression (training-data-corruption guard, self-side twin of the opponent item path above):
        # our OWN current item + ability are request-known ground truth, but the self mon carries no
        # belief entry, so before the fix the self-token item/ability buckets encoded NOTHING —
        # changing p1's item (Choice Band <-> Leftovers) or ability (Immunity <-> Thick Fat) left the
        # entire self token byte-identical, blinding the policy to its own item/ability on every
        # decision (high incidence). The encoder now populates the self fact buckets straight from the
        # request-known current fields (candidate.item / candidate.ability) with zero uncertainty. This
        # is the INVERSE of that reproduction: the self token must now VARY with item and with ability.
        from pokezero.showdown import (
            CATEGORY_BELIEF_ABILITY_OFFSET,
            CATEGORY_BELIEF_ITEM_OFFSET,
            NUMERIC_ACTIVE,
            NUMERIC_POSSIBLE_ABILITY_COUNT,
            NUMERIC_POSSIBLE_ITEM_COUNT,
            NUMERIC_PRESENT,
            NUMERIC_REVEALED_ABILITY,
            NUMERIC_REVEALED_ITEM,
        )

        vocab = build_category_vocabulary(
            [
                "request_kind:move", "stats", "transition:self", "transition:opponent",
                "species:Snorlax", "species:Charizard",
                "belief:possible_item:choiceband", "belief:possible_item:leftovers",
                "belief:possible_ability:immunity", "belief:possible_ability:thickfat",
            ]
        )

        def self_active_token(item, ability):
            # ``item=None`` models a request whose active mon currently holds nothing — the
            # Showdown request empties the item field after Knock Off / Trick / a consumed berry,
            # so this is the self-side "not-currently-held" surface.
            request = {
                "active": [{"moves": [{"id": "bodyslam", "move": "Body Slam"}]}],
                "side": {
                    "id": "p1",
                    "name": "Us",
                    "pokemon": [
                        {
                            "ident": "p1a: Snorlax",
                            "details": "Snorlax, L78",
                            "condition": "100/100",
                            "active": True,
                            "ability": ability,
                            "item": item or "",
                            "moves": ["bodyslam"],
                            "stats": {"atk": 200, "def": 180, "spa": 140, "spd": 200, "spe": 60},
                        }
                    ],
                },
            }
            lines = [
                "|player|p1|Us|",
                "|player|p2|Them|",
                "|switch|p1a: Snorlax|Snorlax, L78|100/100",
                "|switch|p2a: Charizard|Charizard, L78|100/100",
                "|request|" + json.dumps(request),
                "|turn|1",
            ]
            replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
            state = normalize_for_player(replay, player_id="agent", configured_showdown_slot="p1")
            obs = observation_from_player_state(
                state, category_vocab=vocab, spec=V2_1_REPLAY_OBSERVATION_SPEC
            )
            idx = next(
                FIELD_TOKEN_COUNT + i
                for i in range(SELF_POKEMON_TOKEN_COUNT)
                if obs.numeric_features[FIELD_TOKEN_COUNT + i][NUMERIC_ACTIVE] == 1.0
                and obs.numeric_features[FIELD_TOKEN_COUNT + i][NUMERIC_PRESENT] == 1.0
            )
            return obs.numeric_features[idx], obs.categorical_ids[idx]

        # A known item + ability reach the self token with zero uncertainty (singleton bucket).
        cb_num, cb_cat = self_active_token("Choice Band", "Immunity")
        self.assertEqual(cb_num[NUMERIC_REVEALED_ITEM], 1.0)
        self.assertEqual(cb_num[NUMERIC_REVEALED_ABILITY], 1.0)
        self.assertEqual(cb_num[NUMERIC_POSSIBLE_ITEM_COUNT], 1.0)
        self.assertEqual(cb_num[NUMERIC_POSSIBLE_ABILITY_COUNT], 1.0)
        self.assertEqual(cb_cat[CATEGORY_BELIEF_ITEM_OFFSET], vocab.encode("belief:possible_item:choiceband"))
        self.assertEqual(cb_cat[CATEGORY_BELIEF_ABILITY_OFFSET], vocab.encode("belief:possible_ability:immunity"))

        # Inverse of the reproduction: changing ONLY the item moves the item bucket, leaves ability.
        lf_num, lf_cat = self_active_token("Leftovers", "Immunity")
        self.assertEqual(lf_cat[CATEGORY_BELIEF_ITEM_OFFSET], vocab.encode("belief:possible_item:leftovers"))
        self.assertNotEqual(cb_cat[CATEGORY_BELIEF_ITEM_OFFSET], lf_cat[CATEGORY_BELIEF_ITEM_OFFSET])
        self.assertEqual(cb_cat[CATEGORY_BELIEF_ABILITY_OFFSET], lf_cat[CATEGORY_BELIEF_ABILITY_OFFSET])

        # Changing ONLY the ability moves the ability bucket, leaves the item.
        tf_num, tf_cat = self_active_token("Choice Band", "Thick Fat")
        self.assertEqual(tf_cat[CATEGORY_BELIEF_ABILITY_OFFSET], vocab.encode("belief:possible_ability:thickfat"))
        self.assertNotEqual(cb_cat[CATEGORY_BELIEF_ABILITY_OFFSET], tf_cat[CATEGORY_BELIEF_ABILITY_OFFSET])
        self.assertEqual(cb_cat[CATEGORY_BELIEF_ITEM_OFFSET], tf_cat[CATEGORY_BELIEF_ITEM_OFFSET])

        # Ground truth: a self mon whose request shows no held item (Knock Off / consumed berry)
        # encodes not-currently-held — current-held column and count go to 0 and the item bucket is
        # empty — while the still-known ability stays encoded.
        ko_num, ko_cat = self_active_token(None, "Immunity")
        self.assertEqual(ko_num[NUMERIC_REVEALED_ITEM], 0.0)
        self.assertEqual(ko_num[NUMERIC_POSSIBLE_ITEM_COUNT], 0.0)
        self.assertEqual(ko_cat[CATEGORY_BELIEF_ITEM_OFFSET], vocab.encode(""))
        self.assertEqual(ko_num[NUMERIC_REVEALED_ABILITY], 1.0)
        self.assertEqual(ko_cat[CATEGORY_BELIEF_ABILITY_OFFSET], vocab.encode("belief:possible_ability:immunity"))

    def test_self_move_mechanics_id_resolves_hidden_power_variant(self) -> None:
        # Fix 1 unit: the SELF action token must encode Hidden Power's TRUE type/base power, not the
        # generic "hiddenpower" family placeholder (Normal, base power 0). Resolution prefers the
        # authoritative display name, then falls back to the mon's own typed move id (IV-derived by
        # Showdown); every non-HP move — and HP with no resolvable source — passes through.
        for display, expected in (
            ("Hidden Power Fighting 70", "hiddenpowerfighting"),
            ("Hidden Power Ice 70", "hiddenpowerice"),
            ("Hidden Power Grass 70", "hiddenpowergrass"),
            ("Hidden Power Ground 70", "hiddenpowerground"),
        ):
            self.assertEqual(_hidden_power_variant_from_name(display), expected)
        self.assertIsNone(_hidden_power_variant_from_name("Thunderbolt"))
        self.assertIsNone(_hidden_power_variant_from_name(None))
        # Primary: display name.
        self.assertEqual(
            _self_move_mechanics_id({"move": "Hidden Power Ice 70", "id": "hiddenpower"}, "hiddenpower"),
            "hiddenpowerice",
        )
        # Fallback: no usable display name -> the mon's own typed move id.
        self.assertEqual(
            _self_move_mechanics_id(
                {"id": "hiddenpower"}, "hiddenpower", ["thunderbolt", "hiddenpowergrass", "rest"]
            ),
            "hiddenpowergrass",
        )
        # Degenerate (nothing resolvable) -> generic, i.e. unchanged from today.
        self.assertEqual(
            _self_move_mechanics_id({"id": "hiddenpower"}, "hiddenpower", ["thunderbolt"]), "hiddenpower"
        )
        # Non-HP passthrough (Return/Frustration are fixed downstream in dex.resolve_move_base_power).
        self.assertEqual(_self_move_mechanics_id({"move": "Return 102", "id": "return"}, "return"), "return")
        self.assertEqual(_self_move_mechanics_id({"move": "Earthquake", "id": "earthquake"}, "earthquake"), "earthquake")

    @unittest.skipUnless(
        Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown/data/random-battles/gen3/sets.json").exists(),
        "requires a real Gen 3 Showdown checkout for the dex + vocab",
    )
    def test_self_hidden_power_and_return_action_tokens_encode_true_mechanics(self) -> None:
        # Fix 1 integration (training-data-corruption guard): the acting mon's own Hidden Power is on
        # its decision surface every turn (162/220 Gen 3 species). The Showdown request keys the move
        # `id` to the generic "hiddenpower" family whose dex entry is a 0-power, Normal placeholder, so
        # the self action token used to encode base power 0 / type Normal — blinding the policy/value
        # net to its single most common coverage move. Return (happiness-variable, static base power 0)
        # is the same corruption on the power scalar. Post-fix the action token carries the true typed
        # mechanics WHILE the move IDENTITY (CATEGORY_PRIMARY) stays the generic family id, so the fix
        # is checkpoint-compatible (no observation-schema / column-layout change, only values).
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary

        root = "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
        dex = load_showdown_dex_cached(root)
        vocab = gen3_category_vocabulary(root)
        request = json.dumps(
            {
                "active": [
                    {
                        "moves": [
                            {"move": "Hidden Power Ice 70", "id": "hiddenpower"},
                            {"move": "Return 102", "id": "return"},
                            {"move": "Thunderbolt", "id": "thunderbolt"},
                            {"move": "Rest", "id": "rest"},
                        ],
                        "trapped": False,
                    }
                ],
                "side": {
                    "id": "p2",
                    "name": "Them",
                    "pokemon": [
                        {
                            "ident": "p2a: Zapdos",
                            "details": "Zapdos, L78",
                            "condition": "100/100",
                            "active": True,
                            "moves": ["hiddenpowerice", "return", "thunderbolt", "rest"],
                        },
                        {
                            "ident": "p2b: Snorlax",
                            "details": "Snorlax, L78",
                            "condition": "100/100",
                            "active": False,
                        },
                    ],
                },
            }
        )
        lines = [
            "|player|p1|Us|",
            "|player|p2|Them|",
            "|switch|p1a: Starmie|Starmie, L78|100/100",
            "|switch|p2a: Zapdos|Zapdos, L78|100/100",
            "|turn|1",
            "|request|" + request,
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(replay, player_id="agent", configured_showdown_slot="p2")
        obs = observation_from_player_state(
            state, category_vocab=vocab, spec=V2_1_REPLAY_OBSERVATION_SPEC, dex=dex
        )
        action_offset = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT + OPPONENT_POKEMON_TOKEN_COUNT
        hp_cat, hp_num = obs.categorical_ids[action_offset + 0], obs.numeric_features[action_offset + 0]
        ret_cat, ret_num = obs.categorical_ids[action_offset + 1], obs.numeric_features[action_offset + 1]

        # Hidden Power Ice: true type (Ice, not Normal), Special damage class, base power 70 (0.35).
        self.assertEqual(hp_cat[CATEGORY_PRIMARY], vocab.encode("move:hiddenpower"))  # IDENTITY stays generic
        self.assertNotEqual(hp_cat[CATEGORY_TYPE_1], vocab.encode("type:Normal"))
        self.assertEqual(hp_cat[CATEGORY_TYPE_1], vocab.encode("type:Ice"))
        self.assertEqual(hp_cat[CATEGORY_MOVE_CATEGORY], vocab.encode("move_category:Special"))
        self.assertAlmostEqual(hp_num[NUMERIC_BASE_POWER], 70.0 / 200.0, places=6)

        # Return: type/category already correct (Normal/Physical); only the 0 power scalar was wrong.
        self.assertEqual(ret_cat[CATEGORY_PRIMARY], vocab.encode("move:return"))
        self.assertEqual(ret_cat[CATEGORY_TYPE_1], vocab.encode("type:Normal"))
        self.assertAlmostEqual(ret_num[NUMERIC_BASE_POWER], 102.0 / 200.0, places=6)

    @unittest.skipUnless(
        Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown/data/random-battles/gen3/sets.json").exists(),
        "requires a real Gen 3 Showdown checkout for the dex",
    )
    def test_struggle_action_token_encodes_typeless(self) -> None:
        # Gen 3 correctness: Struggle is TYPELESS (neutral vs all types — it HITS Ghosts — and grants
        # no STAB), matching the engine fix (poke-engine-gen3-struggle-typeless.patch). The Showdown
        # dex still records Struggle as Normal-type, which made the SELF forced-Struggle action token
        # encode `type:Normal` — telling the policy/value net that Struggle is Ghost-immune and Rock/
        # Steel-resisted, corrupting PP-exhaustion lines. Post-fix the move-type token is the enumerated
        # typeless `type:???` (the same token gen3 Curse uses). Damage class (Physical) and base power
        # (50) are unchanged, so the fix is value-only / checkpoint-compatible.
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.showdown import (
            CATEGORY_MOVE_CATEGORY,
            _CATEGORICAL_FEATURE_COUNT,
            _V2_1_NUMERIC_FEATURE_COUNT,
        )

        root = "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
        dex = load_showdown_dex_cached(root)

        def encode(move_name: str) -> tuple[str, str, float]:
            cat_row = [""] * _CATEGORICAL_FEATURE_COUNT
            num_row = [0.0] * _V2_1_NUMERIC_FEATURE_COUNT
            _encode_move_mechanics(cat_row, num_row, dex, move_name, user_types=("normal", "typeless"))
            return cat_row[CATEGORY_TYPE_1], cat_row[CATEGORY_MOVE_CATEGORY], num_row[NUMERIC_BASE_POWER]

        struggle_type, struggle_cat, struggle_bp = encode("struggle")
        self.assertEqual(struggle_type, "type:???")
        self.assertNotEqual(struggle_type, "type:Normal")
        # Category + base power unchanged (checkpoint-compatible; only the move-type cell moves).
        self.assertEqual(struggle_cat, "move_category:Physical")
        self.assertAlmostEqual(struggle_bp, 50.0 / 200.0, places=6)
        # Control: a genuine Normal move (Tackle) still encodes type:Normal — only Struggle changed.
        self.assertEqual(encode("tackle")[0], "type:Normal")
        # Struggle mirrors gen3 Curse, whose dex type is already "???".
        self.assertEqual(encode("curse")[0], "type:???")

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
        obs = observation_from_player_state(
            state, category_vocab=vocab, dex=dex, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )

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
            obs = observation_from_player_state(
                state, category_vocab=vocab, dex=dex, spec=V2_1_REPLAY_OBSERVATION_SPEC
            )
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

    @unittest.skipUnless(
        Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown/data/random-battles/gen3/sets.json").exists(),
        "requires a real Gen 3 Showdown checkout for the dex + vocab",
    )
    def test_self_ditto_transform_surfaces_target_identity(self) -> None:
        # SELF-side Ditto: OUR Ditto Transforms into the opponent's Charizard. The self token must
        # surface the COPIED identity (species/types/base stats) exactly like the opponent path,
        # keeping Ditto's own base HP (Transform never copies HP). Pre-fix the self token was stuck
        # on ditto/Normal/48-across because the transform flag lives in the self EXACT belief but the
        # species/type/base-stat surface only consulted the (self-side None) set-source belief. The
        # opponent Charizard token and the benched non-transformed self mon must be unaffected.
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary
        from pokezero.showdown import (
            CATEGORY_PRIMARY, CATEGORY_TYPE_1, CATEGORY_TYPE_2, NUMERIC_ACTIVE,
        )

        root = "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
        dex = load_showdown_dex_cached(root)
        vocab = gen3_category_vocabulary(root)

        def self_request(active_ident: str, active_species: str) -> str:
            # A minimal move-request carrying our two-mon side. A transformed Ditto's request keeps
            # its details=="Ditto" (Showdown never rewrites the species), so candidate.species stays
            # ditto — exactly the case where the copied identity must come from the belief ledger.
            payload = {
                "active": [{"moves": [
                    {"move": "Transform", "id": "transform", "pp": 5, "maxpp": 5,
                     "target": "normal", "disabled": False}
                ]}],
                "side": {"name": "Us", "id": "p1", "pokemon": [
                    {"ident": "p1a: Ditto", "details": "Ditto, L78", "condition": "100/100",
                     "active": active_ident == "Ditto",
                     "stats": {"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
                     "moves": ["transform"], "baseAbility": "limber", "item": "quickclaw",
                     "ability": "limber"},
                    {"ident": "p1: Swampert", "details": "Swampert, L78", "condition": "100/100",
                     "active": active_ident == "Swampert",
                     "stats": {"atk": 150, "def": 150, "spa": 130, "spd": 130, "spe": 100},
                     "moves": ["surf"], "baseAbility": "torrent", "item": "leftovers",
                     "ability": "torrent"},
                ]},
            }
            return "|request|" + json.dumps(payload)

        def self_tokens(lines):
            replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
            state = normalize_for_player(replay, player_id="agent", configured_showdown_slot="p1")
            obs = observation_from_player_state(
                state, category_vocab=vocab, dex=dex, spec=V2_1_REPLAY_OBSERVATION_SPEC
            )
            self_off = FIELD_TOKEN_COUNT
            num = [obs.numeric_features[self_off + i] for i in range(SELF_POKEMON_TOKEN_COUNT)]
            cat = [obs.categorical_ids[self_off + i] for i in range(SELF_POKEMON_TOKEN_COUNT)]
            opp_off = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT
            opp_num = [obs.numeric_features[opp_off + i] for i in range(OPPONENT_POKEMON_TOKEN_COUNT)]
            opp_cat = [obs.categorical_ids[opp_off + i] for i in range(OPPONENT_POKEMON_TOKEN_COUNT)]
            return num, cat, opp_num, opp_cat

        base = [
            "|player|p1|Us|",
            "|player|p2|Them|",
            "|switch|p1a: Ditto|Ditto, L78|100/100",
            "|switch|p2a: Charizard|Charizard, L78|100/100",
            "|turn|1",
            "|move|p1a: Ditto|Transform|p2a: Charizard",
            "|-transform|p1a: Ditto|p2a: Charizard",
            self_request("Ditto", "Ditto"),
            "|turn|2",
        ]

        def active_row(num, cat):
            idx = next(i for i, row in enumerate(num) if row[NUMERIC_ACTIVE] == 1.0)
            return num[idx], cat[idx], idx

        # --- FIRST transform: self Ditto fights as Charizard. ---
        num, cat, opp_num, opp_cat = self_tokens(base)
        srow, scat, sidx = active_row(num, cat)
        self.assertEqual(scat[CATEGORY_PRIMARY], vocab.encode("species:Charizard"))
        self.assertEqual(scat[CATEGORY_TYPE_1], vocab.encode("type:Fire"))
        self.assertEqual(scat[CATEGORY_TYPE_2], vocab.encode("type:Flying"))
        self.assertAlmostEqual(srow[NUMERIC_BASE_ATK], 84 / 200)   # Charizard (copied)
        self.assertAlmostEqual(srow[NUMERIC_BASE_DEF], 78 / 200)
        self.assertAlmostEqual(srow[NUMERIC_BASE_SPA], 109 / 200)
        self.assertAlmostEqual(srow[NUMERIC_BASE_SPD], 85 / 200)
        self.assertAlmostEqual(srow[NUMERIC_BASE_SPE], 100 / 200)
        self.assertAlmostEqual(srow[NUMERIC_BASE_HP], 48 / 200)    # Ditto's HP (NOT copied)
        # The benched, non-transformed self mon (Swampert) is unaffected.
        swampert_idx = next(
            i for i, row in enumerate(cat)
            if row[CATEGORY_PRIMARY] == vocab.encode("species:Swampert")
        )
        self.assertNotEqual(swampert_idx, sidx)
        self.assertAlmostEqual(num[swampert_idx][NUMERIC_BASE_ATK], 110 / 200)  # Swampert's own
        # The opponent Charizard token is unchanged (the opponent path already worked).
        opp_idx = next(i for i, row in enumerate(opp_num) if row[NUMERIC_ACTIVE] == 1.0)
        self.assertEqual(opp_cat[opp_idx][CATEGORY_PRIMARY], vocab.encode("species:Charizard"))
        self.assertEqual(opp_cat[opp_idx][CATEGORY_TYPE_1], vocab.encode("type:Fire"))

    @unittest.skipUnless(
        Path("/Users/scott/workspace/pokerena/vendor/pokemon-showdown/data/random-battles/gen3/sets.json").exists(),
        "requires a real Gen 3 Showdown checkout for the dex + vocab",
    )
    def test_self_ditto_retransform_after_switch(self) -> None:
        # Codex's full chain: OUR Ditto Transforms, switches out (reverts), switches back in, and
        # Transforms AGAIN. The switch-out reset must clear the copied identity and the re-transform
        # must re-apply it — every transformed self decision, not just the first.
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary
        from pokezero.showdown import CATEGORY_PRIMARY, CATEGORY_TYPE_1, NUMERIC_ACTIVE

        root = "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
        dex = load_showdown_dex_cached(root)
        vocab = gen3_category_vocabulary(root)

        def self_request(active: str) -> str:
            payload = {
                "active": [{"moves": [
                    {"move": "Transform" if active == "Ditto" else "Surf",
                     "id": "transform" if active == "Ditto" else "surf",
                     "pp": 5, "maxpp": 5, "target": "normal", "disabled": False}
                ]}],
                "side": {"name": "Us", "id": "p1", "pokemon": [
                    {"ident": "p1a: Ditto", "details": "Ditto, L78", "condition": "100/100",
                     "active": active == "Ditto",
                     "stats": {"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
                     "moves": ["transform"], "baseAbility": "limber", "item": "quickclaw",
                     "ability": "limber"},
                    {"ident": "p1: Swampert", "details": "Swampert, L78", "condition": "100/100",
                     "active": active == "Swampert",
                     "stats": {"atk": 150, "def": 150, "spa": 130, "spd": 130, "spe": 100},
                     "moves": ["surf"], "baseAbility": "torrent", "item": "leftovers",
                     "ability": "torrent"},
                ]},
            }
            return "|request|" + json.dumps(payload)

        def self_active(lines):
            replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
            state = normalize_for_player(replay, player_id="agent", configured_showdown_slot="p1")
            obs = observation_from_player_state(
                state, category_vocab=vocab, dex=dex, spec=V2_1_REPLAY_OBSERVATION_SPEC
            )
            self_off = FIELD_TOKEN_COUNT
            num = [obs.numeric_features[self_off + i] for i in range(SELF_POKEMON_TOKEN_COUNT)]
            cat = [obs.categorical_ids[self_off + i] for i in range(SELF_POKEMON_TOKEN_COUNT)]
            idx = next(i for i, row in enumerate(num) if row[NUMERIC_ACTIVE] == 1.0)
            return num[idx], cat[idx]

        transform1 = [
            "|player|p1|Us|",
            "|player|p2|Them|",
            "|switch|p1a: Ditto|Ditto, L78|100/100",
            "|switch|p2a: Charizard|Charizard, L78|100/100",
            "|turn|1",
            "|move|p1a: Ditto|Transform|p2a: Charizard",
            "|-transform|p1a: Ditto|p2a: Charizard",
        ]
        # Ditto pivots out to Swampert (reverts on the bench), then Swampert pivots back to Ditto.
        reverted = transform1 + [
            "|turn|2",
            "|switch|p1a: Swampert|Swampert, L78|100/100",
            "|turn|3",
            "|switch|p1a: Ditto|Ditto, L78|100/100",
            self_request("Ditto"),
            "|turn|4",
        ]
        retransformed = reverted[:-1] + [
            "|move|p1a: Ditto|Transform|p2a: Charizard",
            "|-transform|p1a: Ditto|p2a: Charizard",
            self_request("Ditto"),
            "|turn|4",
        ]

        # After switch-back-in, before the second Transform: reverted to plain Ditto.
        rnum, rcat = self_active(reverted)
        self.assertEqual(rcat[CATEGORY_PRIMARY], vocab.encode("species:Ditto"))
        self.assertEqual(rcat[CATEGORY_TYPE_1], vocab.encode("type:Normal"))
        self.assertAlmostEqual(rnum[NUMERIC_BASE_ATK], 48 / 200)  # Ditto's own, not Charizard's

        # Re-transform: the copied Charizard identity is surfaced again.
        tnum, tcat = self_active(retransformed)
        self.assertEqual(tcat[CATEGORY_PRIMARY], vocab.encode("species:Charizard"))
        self.assertEqual(tcat[CATEGORY_TYPE_1], vocab.encode("type:Fire"))
        self.assertAlmostEqual(tnum[NUMERIC_BASE_ATK], 84 / 200)  # Charizard (copied)
        self.assertAlmostEqual(tnum[NUMERIC_BASE_HP], 48 / 200)   # Ditto's HP (NOT copied)

    def test_revealed_moves_survive_bucket_truncation(self) -> None:
        # A revealed (ground-truth) move must never be evicted by the encoder's alphabetical
        # sort+truncate, even when possible_moves alone would overflow the 16 buckets and the
        # revealed move sorts last. (Off-script: the revealed move is not among possible_moves.)
        from pokezero.showdown import _prioritized_belief_moves, _normalize_identifier

        revealed = ("Zap Cannon",)  # sorts after any "aaa..." possible move
        possible = tuple(f"aaamove{i:02d}" for i in range(20))  # 20 > 16 buckets
        result = _prioritized_belief_moves(revealed, possible, 16)

        self.assertIn("Zap Cannon", result)
        self.assertLessEqual(len({_normalize_identifier(m) for m in result}), 16)

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
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )

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
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )

        self.assertEqual(state.self_side_conditions, ("reflect", "spikes"))
        self.assertEqual(state.opponent_side_conditions, ("spikes",))
        self.assertEqual(state.self_side_condition_counts, {"reflect": 1, "spikes": 1})
        self.assertEqual(state.opponent_side_condition_counts, {"spikes": 3})
        self.assertEqual(observation.metadata["self_side_condition_counts"], {"reflect": 1, "spikes": 1})
        self.assertEqual(observation.metadata["opponent_side_condition_counts"], {"spikes": 3})

    def test_observation_encodes_player_relative_content(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
        # Spec v2 tail: one stats token, then the transition-token block (chronological,
        # zero-padded, attention-masked beyond the filled slots).
        stats_offset = event_offset
        transition_offset = stats_offset + STATS_TOKEN_COUNT
        self.assertEqual(observation.categorical_ids[stats_offset][2], stable_category_id("stats"))
        self.assertTrue(observation.attention_mask[stats_offset])
        # Fixture history: p1 lead switch, p1 voluntary switch, p2 lead switch, then two moves.
        self.assertEqual(len(state.transition_tokens), 5)
        self.assertEqual(
            observation.categorical_ids[transition_offset][3], stable_category_id("tt_kind:switch")
        )
        move_token = transition_offset + 3
        self.assertEqual(observation.categorical_ids[move_token][0], stable_category_id("species:Charizard"))
        self.assertEqual(observation.categorical_ids[move_token][1], stable_category_id("move:flamethrower"))
        self.assertEqual(observation.categorical_ids[move_token][2], stable_category_id("transition:self"))
        self.assertEqual(observation.categorical_ids[move_token][3], stable_category_id("tt_kind:move"))
        self.assertEqual(observation.categorical_ids[move_token][4], stable_category_id("tt_outcome:normal"))
        self.assertAlmostEqual(
            observation.numeric_features[move_token][NUMERIC_TT_DAMAGE_FRACTION], 0.3, places=6
        )
        opp_move_token = transition_offset + 4
        self.assertEqual(
            observation.categorical_ids[opp_move_token][2], stable_category_id("transition:opponent")
        )
        self.assertTrue(all(observation.attention_mask[transition_offset : transition_offset + 5]))
        self.assertFalse(any(observation.attention_mask[transition_offset + 5 :]))

    def test_observation_encodes_public_belief_summary_features(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(
            replay,
            player_id="agent",
            player_name="PokeZeroBot",
            format_id="gen3randombattle",
            set_source=FakeSetSource(),
        )

        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
        reordered = observation_from_player_state(
            reordered_state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
            state, category_vocab=_TEST_VOCAB, dex=_phase2_fake_dex(),
            spec=V2_1_REPLAY_OBSERVATION_SPEC,
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
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        numeric = observation.numeric_features[self.SELF_ACTIVE_OFFSET]
        self.assertAlmostEqual(numeric[NUMERIC_LEVEL], 0.78)  # parsed from details, no dex needed
        self.assertEqual(numeric[NUMERIC_BASE_SPA], 0.0)  # base stats need the dex

    def test_weather_parsed_and_encoded_on_field_token(self) -> None:
        state = self._replay_with(["|-weather|RainDance"])
        self.assertEqual(state.weather, "raindance")
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        self.assertEqual(
            observation.categorical_ids[FIELD_TOKEN_OFFSET][CATEGORY_SECONDARY],
            stable_category_id("weather:raindance"),
        )

    def test_weather_none_clears(self) -> None:
        state = self._replay_with(["|-weather|RainDance", "|-weather|none"])
        self.assertIsNone(state.weather)

    def _encoded_weather(self, extra_lines: list[str]) -> tuple[float, float]:
        """(weather_turns column, weather_permanent column) for the injected weather lines."""
        state = self._replay_with(extra_lines)
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        field = observation.numeric_features[FIELD_TOKEN_OFFSET]
        return field[NUMERIC_WEATHER_TURNS], field[NUMERIC_WEATHER_PERMANENT]

    def test_move_weather_countdown_consumes_set_turn_upkeep_at_first_boundary(self) -> None:
        # Audit #9: the first post-resolution decision after a fresh move-weather line already
        # sits AFTER the set turn's end-of-turn |-weather|...|[upkeep] tick, so it must encode
        # four remaining turns (matching the bridge weatherState.duration of 4), not the full five.
        start = parse_showdown_replay(["|-weather|RainDance", "|-weather|RainDance|[upkeep]"])
        self.assertEqual(_weather_duration_features(start), (4, False))
        self.assertAlmostEqual(self._encoded_weather(
            ["|-weather|RainDance", "|-weather|RainDance|[upkeep]"]
        )[0], 4 / 5)
        # One later boundary: a second upkeep tick -> three remaining (bridge duration 3).
        later = parse_showdown_replay(
            ["|-weather|RainDance", "|-weather|RainDance|[upkeep]", "|turn|2",
             "|-weather|RainDance|[upkeep]"]
        )
        self.assertEqual(_weather_duration_features(later), (3, False))

    def test_move_weather_before_any_upkeep_is_full_five(self) -> None:
        # Before the set turn's upkeep fires (e.g. a mid-turn forced switch on the set turn), the
        # bridge still reports the full duration 5 -- the fix must NOT over-consume here.
        fresh = parse_showdown_replay(["|-weather|RainDance"])
        self.assertEqual(_weather_duration_features(fresh), (5, False))
        self.assertAlmostEqual(self._encoded_weather(["|-weather|RainDance"])[0], 5 / 5)

    def test_move_weather_recast_countdown_after_expiry(self) -> None:
        # A second Rain Dance after the first expires must restart the countdown and again consume
        # the recast set turn's upkeep at the first post-resolution boundary (bridge duration 4).
        recast_prefix = [
            "|-weather|RainDance", "|-weather|RainDance|[upkeep]", "|turn|2",
            "|-weather|RainDance|[upkeep]", "|turn|3", "|-weather|RainDance|[upkeep]", "|turn|4",
            "|-weather|RainDance|[upkeep]", "|turn|5", "|-weather|none", "|turn|6",
            "|-weather|RainDance", "|-weather|RainDance|[upkeep]",
        ]
        recast_first = parse_showdown_replay(recast_prefix)
        self.assertEqual(recast_first.weather, "raindance")
        self.assertEqual(_weather_duration_features(recast_first), (4, False))
        # One later recast boundary -> three remaining (bridge duration 3).
        recast_later = parse_showdown_replay(
            recast_prefix + ["|turn|7", "|-weather|RainDance|[upkeep]"]
        )
        self.assertEqual(_weather_duration_features(recast_later), (3, False))

    def test_permanent_ability_weather_is_unchanged_by_upkeeps(self) -> None:
        # Gen 3 ability weather (Sand Stream) is permanent (bridge duration 0). The upkeep counter
        # must never bleed into its encoding: it stays pinned at the full five and flagged permanent
        # across any number of upkeep ticks -- byte-identical to the pre-fix short-circuit.
        for upkeeps in range(4):
            lines = ["|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar"]
            lines += ["|-weather|Sandstorm|[upkeep]"] * upkeeps
            state = parse_showdown_replay(lines)
            self.assertEqual(_weather_duration_features(state), (5, True))
        turns, permanent = self._encoded_weather(
            ["|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar",
             "|-weather|Sandstorm|[upkeep]", "|-weather|Sandstorm|[upkeep]"]
        )
        self.assertAlmostEqual(turns, 5 / 5)
        self.assertAlmostEqual(permanent, 1.0)

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
            state, category_vocab=_TEST_VOCAB, dex=_phase2_fake_dex(),
            spec=V2_1_REPLAY_OBSERVATION_SPEC,
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

    def test_failed_baton_pass_does_not_leak_to_a_later_switch_or_turn(self) -> None:
        failed = parse_showdown_replay(
            [
                "|move|p2a: Charizard|Baton Pass",
                "|-fail|p2a: Charizard|move: Baton Pass",
            ]
        )
        self.assertEqual(failed.pending_baton_pass, ())

        stale_turn = parse_showdown_replay(
            [
                "|move|p2a: Charizard|Baton Pass",
                "|turn|8",
            ]
        )
        self.assertEqual(stale_turn.pending_baton_pass, ())

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

    def test_baton_pass_carries_public_volatiles_and_marks_direct_state_gaps(self) -> None:
        replay = parse_showdown_replay(
            [
                "|-start|p2a: Charizard|move: Ingrain",
                "|-start|p2a: Charizard|Substitute",
                "|move|p2a: Charizard|Baton Pass",
                "|switch|p2a: Snorlax|Snorlax, L78|100/100",
            ],
            battle_id="battle-gen3randombattle-1",
        )

        # Both effects copy in Gen 3. Ingrain has a complete direct-state payload, while
        # Substitute's private remaining HP makes direct materialization fail closed.
        self.assertEqual(replay.volatiles["p2"], ("ingrain", "substitute"))
        self.assertEqual(replay.direct_materialization_blockers["p2"], ("baton-pass:substitute",))

        cleared = parse_showdown_replay(
            [
                "|-start|p2a: Charizard|Substitute",
                "|move|p2a: Charizard|Baton Pass",
                "|switch|p2a: Snorlax|Snorlax, L78|100/100",
                "|switch|p2a: Charizard|Charizard, L78|100/100",
            ],
            battle_id="battle-gen3randombattle-1",
        )
        self.assertEqual(cleared.volatiles["p2"], ())
        self.assertEqual(cleared.direct_materialization_blockers["p2"], ())

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
            state, category_vocab=_TEST_VOCAB, dex=_phase2_fake_dex(),
            spec=V2_1_REPLAY_OBSERVATION_SPEC,
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
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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

    @staticmethod
    def _tox_pivot_lines(mon: str, *, extra: list[str] | None = None) -> list[str]:
        """A badly-poisoned mon (active-slot prefix ``mon``, e.g. 'p1a') that escalates, pivots
        OUT (counter reset to 0, Gen 3) and back IN carrying ``tox`` only in the switch condition
        with no fresh ``|-status|`` — mirrors the live Tauros/Milotic capture. Gen 3 residual damage
        is ``max(1, floor(285/16)) * stage`` = ``17 * stage`` (17, 34, 51 → stage 1, 2, 3)."""
        foe = "p2a" if mon.startswith("p1") else "p1a"
        return [
            f"|switch|{mon}: Tauros|Tauros, L80, M|285/285",
            f"|switch|{foe}: Milotic|Milotic, L80, F|317/317",
            "|turn|1",
            f"|-status|{mon}: Tauros|tox",
            f"|-damage|{mon}: Tauros|268/285 tox|[from] psn",  # 17 = stage 1
            "|upkeep",
            "|turn|2",
            f"|-damage|{mon}: Tauros|234/285 tox|[from] psn",  # 34 = stage 2
            "|upkeep",
            "|turn|3",
            f"|switch|{mon}: Zapdos|Zapdos, L78|301/301",  # Tauros leaves: counter reset to 0
            "|upkeep",
            "|turn|4",
            f"|switch|{mon}: Tauros|Tauros, L80, M|234/285 tox",  # RE-ENTRY, no |-status|
            f"|-damage|{mon}: Tauros|217/285 tox|[from] psn",  # 17 = stage 1 RESTART (re-seed)
            "|upkeep",
            "|turn|5",
            *(extra or []),
        ]

    def test_toxic_ramp_reseeds_from_public_damage_after_pivot_both_seats(self) -> None:
        # After a pivot the ramp is re-derived from the PUBLIC end-of-turn residual (the exact
        # counter is hidden but round(16 * damage/maxhp) recovers it), so a re-entered tox mon
        # shows the true escalating stage instead of a stuck 0. Tracked identically for both seats.
        for mon, side in (("p1a", "p1"), ("p2a", "p2")):
            with self.subTest(seat=side):
                # At the turn-5 decision point the mon has re-entered and taken one residual: the
                # ramp restarted at stage 1 (turn 4 damage) and the |turn|5 escalation lifts it to
                # the stage 2 that lands this turn — where the pre-fix encoder read a stuck 0.
                at5 = parse_showdown_replay(self._tox_pivot_lines(mon))
                self.assertEqual(at5.toxic_stage[side], 2)
                # And it keeps climbing: one more residual (34 = stage 2) + escalation → stage 3.
                at6 = parse_showdown_replay(
                    self._tox_pivot_lines(
                        mon,
                        extra=[
                            f"|-damage|{mon}: Tauros|183/285 tox|[from] psn",  # 34 = stage 2
                            "|upkeep",
                            "|turn|6",
                        ],
                    )
                )
                self.assertEqual(at6.toxic_stage[side], 3)

    def test_toxic_ramp_reseed_encodes_correct_stage_on_active_token(self) -> None:
        # End-to-end proof the re-seed reaches the observation column: encode from the FOE's seat
        # (p1 tracking p2's re-entered Tauros) and read NUMERIC_TOXIC_STAGE off its active token.
        from pokezero.showdown import NUMERIC_ACTIVE

        replay = parse_showdown_replay(self._tox_pivot_lines("p2a"))
        state = normalize_for_player(replay, player_id="agent", configured_showdown_slot="p1")
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        opponent_offset = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT
        opp_active = next(
            opponent_offset + i
            for i in range(OPPONENT_POKEMON_TOKEN_COUNT)
            if observation.numeric_features[opponent_offset + i][NUMERIC_ACTIVE] == 1.0
        )
        self.assertAlmostEqual(
            observation.numeric_features[opp_active][NUMERIC_TOXIC_STAGE], 2 / 15
        )

    def test_regular_poison_residual_never_seeds_toxic_ramp(self) -> None:
        # A regular-poison (`psn`, flat 1/8) residual also emits `[from] psn`, but its condition
        # carries no `tox` token, so it must NOT touch the toxic ramp (no false stage from 1/8).
        state = parse_showdown_replay(
            [
                "|switch|p1a: Tauros|Tauros, L80, M|285/285",
                "|switch|p2a: Milotic|Milotic, L80, F|317/317",
                "|turn|1",
                "|-status|p1a: Tauros|psn",
                "|-damage|p1a: Tauros|250/285 psn|[from] psn",  # 35 = 1/8, regular poison
                "|upkeep",
                "|turn|2",
            ]
        )
        self.assertEqual(state.toxic_stage["p1"], 0)

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
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
        volatile_cols = observation.categorical_ids[self.SELF_ACTIVE_OFFSET][
            CATEGORY_VOLATILE_OFFSET : CATEGORY_VOLATILE_OFFSET + 6
        ]
        self.assertIn(stable_category_id("volatile:confusion"), volatile_cols)
        self.assertIn(stable_category_id("volatile:leechseed"), volatile_cols)

    def test_leech_seed_tracks_public_source_side_and_fails_closed_without_one(self) -> None:
        seeded = parse_showdown_replay(
            [
                "|move|p1a: Bulbasaur|Leech Seed|p2a: Charizard",
                "|-start|p2a: Charizard|move: Leech Seed",
            ]
        )
        self.assertEqual(seeded.leech_seed_source_sides, {"p2": "p1"})
        self.assertEqual(seeded.direct_materialization_blockers["p2"], ())

        unknown_source = parse_showdown_replay(
            ["|-start|p2a: Charizard|move: Leech Seed"]
        )
        self.assertEqual(unknown_source.leech_seed_source_sides, {})
        self.assertEqual(
            unknown_source.direct_materialization_blockers["p2"],
            ("leechseed-source-unknown",),
        )

        cleared = parse_showdown_replay(
            [
                "|move|p1a: Bulbasaur|Leech Seed|p2a: Charizard",
                "|-start|p2a: Charizard|move: Leech Seed",
                "|switch|p2a: Snorlax|Snorlax, L78|100/100",
            ]
        )
        self.assertEqual(cleared.leech_seed_source_sides, {})

        baton_passed = parse_showdown_replay(
            [
                "|move|p1a: Bulbasaur|Leech Seed|p2a: Charizard",
                "|-start|p2a: Charizard|move: Leech Seed",
                "|move|p2a: Charizard|Baton Pass|p2a: Charizard",
                "|switch|p2a: Snorlax|Snorlax, L78|100/100|[from] Baton Pass",
            ]
        )
        self.assertEqual(baton_passed.leech_seed_source_sides, {"p2": "p1"})
        self.assertEqual(baton_passed.direct_materialization_blockers["p2"], ())

        unknown_baton_passed = parse_showdown_replay(
            [
                "|-start|p2a: Charizard|move: Leech Seed",
                "|move|p2a: Charizard|Baton Pass|p2a: Charizard",
                "|switch|p2a: Snorlax|Snorlax, L78|100/100|[from] Baton Pass",
            ]
        )
        self.assertEqual(unknown_baton_passed.leech_seed_source_sides, {})
        self.assertEqual(
            unknown_baton_passed.direct_materialization_blockers["p2"],
            ("leechseed-source-unknown",),
        )

    def test_replay_snapshot_restore_preserves_pending_public_conditions(self) -> None:
        parser = _ReplayParser()
        parser.feed(
            [
                "|turn|7",
                "|move|p1a: Jirachi|Wish|p1a: Jirachi",
                "|move|p2a: Bulbasaur|Leech Seed|p1a: Jirachi",
                "|-start|p1a: Jirachi|move: Leech Seed",
                "|-boost|p1a: Jirachi|atk|2",
                "|move|p1a: Jirachi|Baton Pass",
            ]
        )
        snapshot = parser.snapshot()

        restored_parser = _ReplayParser.from_snapshot(snapshot)
        self.assertEqual(restored_parser.snapshot(), snapshot)
        self.assertEqual(snapshot.wish_set_turns, {"p1": 7})
        self.assertEqual(snapshot.leech_seed_source_sides, {"p1": "p2"})
        self.assertEqual(snapshot.pending_baton_pass, ("p1",))

        restored_parser.feed(["|switch|p1a: Snorlax|Snorlax, L78|100/100"])
        restored = restored_parser.snapshot()
        self.assertEqual(restored.boosts["p1"], {"atk": 2})
        self.assertEqual(restored.pending_baton_pass, ())

    def test_snapshot_between_leech_seed_move_and_start_converges_with_live(self) -> None:
        # Snapshot-vs-live convergence for the *inter-message* window: a snapshot taken after the
        # |move| Leech Seed (which records the pending source) but before its matching |-start|
        # (which consumes it) must restore the pending source, so the -start attributes the seed
        # to p1 rather than falling to leechseed-source-unknown. Regression for the pending-map
        # serialization gap (the map is transient and empty at decision boundaries, so this window
        # is not hit in the live encode path, but snapshot-vs-live must still converge).
        move_line = "|move|p1a: Bulbasaur|Leech Seed|p2a: Charizard"
        start_line = "|-start|p2a: Charizard|move: Leech Seed"

        parser = _ReplayParser()
        parser.feed([move_line])
        mid = parser.snapshot()
        self.assertEqual(mid.pending_leech_seed_source_sides, {"p2": "p1"})

        restored = _ReplayParser.from_snapshot(mid)
        restored.feed([start_line])
        restored_state = restored.snapshot()

        live = parse_showdown_replay([move_line, start_line])

        self.assertEqual(restored_state.leech_seed_source_sides, {"p2": "p1"})
        self.assertEqual(restored_state.direct_materialization_blockers["p2"], ())
        self.assertEqual(
            restored_state.leech_seed_source_sides, live.leech_seed_source_sides
        )
        self.assertEqual(
            restored_state.direct_materialization_blockers["p2"],
            live.direct_materialization_blockers["p2"],
        )

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

    def test_partial_trap_volatile_via_activate_and_end(self) -> None:
        # Audit bug C2: the sim announces partial traps via |-activate|...|move: Wrap|
        # (conditions.ts partiallytrapped.onStart) and ends them with the MOVE name +
        # [partiallytrapped] — real lines lifted from the zero-activation audit logs.
        wrapped = self._replay_with(
            ["|-activate|p2a: Charizard|move: Wrap|[of] p1a: Shuckle"]
        )
        self.assertEqual(wrapped.self_active_volatiles, ("partiallytrapped",))
        ended = self._replay_with(
            [
                "|-activate|p2a: Charizard|move: Wrap|[of] p1a: Shuckle",
                "|-end|p2a: Charizard|Wrap|[partiallytrapped]",
            ]
        )
        self.assertEqual(ended.self_active_volatiles, ())
        silent = self._replay_with(
            [
                "|-activate|p2a: Charizard|move: Wrap|[of] p1a: Shuckle",
                "|-end|p2a: Charizard|Wrap|[partiallytrapped]|[silent]",
            ]
        )
        self.assertEqual(silent.self_active_volatiles, ())
        # Pursuit's interception marker is also an |-activate|move:| line — never a trap.
        pursuit = self._replay_with(
            ["|-activate|p2a: Charizard|move: Pursuit"]
        )
        self.assertEqual(pursuit.self_active_volatiles, ())

    def _destiny_bond_replay(self, extra_lines: list[str]):
        # Standalone replay (not _replay_with: its fixture tail appends further
        # Charizard moves, which CORRECTLY expire an armed bond — these tests need to
        # control exactly which action follows the arming line).
        lines = [
            "|player|p1|Other|",
            "|player|p2|PokeZeroBot|",
            "|switch|p1a: Snorlax|Snorlax, L78|100/100",
            "|switch|p2a: Charizard|Charizard, L78|100/100",
            "|turn|1",
            "|move|p2a: Charizard|Destiny Bond|p2a: Charizard",
            "|-singlemove|p2a: Charizard|Destiny Bond",
            *extra_lines,
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
        return normalize_for_player(replay, player_id="agent", player_name="PokeZeroBot")

    def test_destiny_bond_volatile_via_singlemove_expires_on_next_action(self) -> None:
        # Audit bug C3: Destiny Bond arms via |-singlemove| (which the parser ignored)
        # and the sim removes it SILENTLY before the mon's next move (onBeforeMove) or
        # aborted move (onMoveAborted) — no protocol removal line exists.
        armed = self._destiny_bond_replay([])
        self.assertEqual(armed.self_active_volatiles, ("destinybond",))
        moved = self._destiny_bond_replay(
            ["|turn|2", "|move|p2a: Charizard|Flamethrower|p1a: Snorlax"]
        )
        self.assertEqual(moved.self_active_volatiles, ())
        aborted = self._destiny_bond_replay(["|turn|2", "|cant|p2a: Charizard|par"])
        self.assertEqual(aborted.self_active_volatiles, ())
        # A successful re-click re-arms: the |move| clears, the |-singlemove| re-adds.
        rearmed = self._destiny_bond_replay(
            [
                "|turn|2",
                "|move|p2a: Charizard|Destiny Bond|p2a: Charizard",
                "|-singlemove|p2a: Charizard|Destiny Bond",
            ]
        )
        self.assertEqual(rearmed.self_active_volatiles, ("destinybond",))
        # The opponent's move never expires OUR bond.
        theirs = self._destiny_bond_replay(
            ["|turn|2", "|move|p1a: Snorlax|Body Slam|p2a: Charizard"]
        )
        self.assertEqual(theirs.self_active_volatiles, ("destinybond",))

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
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )
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
        observation = observation_from_player_state(
            state, category_vocab=_TEST_VOCAB, spec=V2_1_REPLAY_OBSERVATION_SPEC
        )

        self_active = FIELD_TOKEN_COUNT  # token 1: active Charizard
        self_bench = FIELD_TOKEN_COUNT + 1  # token 2: Snorlax
        opp_active = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT  # token 7: Xatu (no actual stats)
        self.assertAlmostEqual(observation.numeric_features[self_active][NUMERIC_ACTUAL_SPE], 236 / 714)
        self.assertAlmostEqual(observation.numeric_features[self_active][NUMERIC_ACTUAL_HP], 250 / 714)
        self.assertAlmostEqual(observation.numeric_features[self_bench][NUMERIC_ACTUAL_SPE], 90 / 714)
        # The opponent's actual stats are hidden -> the slots stay padding (0).
        self.assertEqual(observation.numeric_features[opp_active][NUMERIC_ACTUAL_SPE], 0.0)
        self.assertEqual(observation.numeric_features[opp_active][NUMERIC_ACTUAL_HP], 0.0)


_SHOWDOWN_ROOT = "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"
_HAS_SHOWDOWN_CHECKOUT = Path(
    _SHOWDOWN_ROOT + "/data/random-battles/gen3/sets.json"
).exists()


class StaticDataTrainingFixTest(unittest.TestCase):
    """Regression tests for two checkpoint-compatible (value-only) static-data encoder fixes.

    F1 — Showdown OMITS the level token from a mon's details string when — and only when —
    the level is 100 (``sim/pokemon.ts::getUpdatedDetails``: ``level === 100 ? '' :
    `, L${level}```), so a token-less details string means L100, not "unknown". The encoder
    must then set NUMERIC_LEVEL=1.0 (self, opponent, and switch tokens) and compute the
    opponent expected-stats block instead of leaving it zeroed.

    F2 — gen3 randbats emit Unown as lettered cosmetic formes (Unown-C, ...) absent from the
    Pokedex; the encoder must fall back to the base species for types + base stats, WITHOUT
    collapsing real distinct dex formes (Deoxys-Attack/Defense/Speed, Castform).
    """

    def test_level_from_details_defaults_missing_token_to_l100(self) -> None:
        from pokezero.showdown import _level_from_details

        # No ", L<n>" token in a non-empty details string => level 100 (Showdown omits it).
        self.assertEqual(_level_from_details("Nosepass, F"), 100)
        self.assertEqual(_level_from_details("Unown-C"), 100)
        self.assertEqual(_level_from_details("Ditto"), 100)
        # A present level token is still parsed exactly (sub-100 unchanged).
        self.assertEqual(_level_from_details("Charizard, L83, M"), 83)
        self.assertEqual(_level_from_details("Zapdos, L78"), 78)
        # Genuinely-absent details carry no level information => None (no default applied).
        self.assertIsNone(_level_from_details(""))
        self.assertIsNone(_level_from_details(None))

    @unittest.skipUnless(_HAS_SHOWDOWN_CHECKOUT, "requires a real Gen 3 Showdown checkout for the dex")
    def test_l100_level_and_expected_stats_with_sub100_unchanged(self) -> None:
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.showdown import (
            NUMERIC_BASE_HP,
            NUMERIC_EXPECTED_ATK,
            NUMERIC_EXPECTED_DEF,
            NUMERIC_EXPECTED_HP,
            NUMERIC_EXPECTED_SPA,
            NUMERIC_EXPECTED_SPD,
            NUMERIC_EXPECTED_SPE,
            _ACTUAL_STAT_DIVISOR,
            _encode_expected_stats,
            _encode_pokemon_stats,
            _gen3_stat,
        )

        dex = load_showdown_dex_cached(_SHOWDOWN_ROOT)
        width = 160

        # F1: an L100 mon (details carries no level token) encodes NUMERIC_LEVEL=1.0 + base stats.
        # _encode_pokemon_stats is the SINGLE path for self, opponent AND switch tokens, so this
        # covers the level column on every seat.
        row = [0.0] * width
        _encode_pokemon_stats(row, dex, "Nosepass", "Nosepass, F")
        self.assertEqual(row[NUMERIC_LEVEL], 1.0)
        self.assertAlmostEqual(row[NUMERIC_BASE_HP], dex.species_info("Nosepass").base_stats["hp"] / 200)

        # Sub-100 mon UNCHANGED: the present level token is parsed, no default applied.
        row = [0.0] * width
        _encode_pokemon_stats(row, dex, "Zapdos", "Zapdos, L78")
        self.assertAlmostEqual(row[NUMERIC_LEVEL], 0.78)

        # F1: the L100 opponent expected-stats block is now computed (was zeroed by an early
        # `level is None` bail). Without a set source the bounds collapse to the baseline spread.
        exp_slots = (
            NUMERIC_EXPECTED_HP,
            NUMERIC_EXPECTED_ATK,
            NUMERIC_EXPECTED_DEF,
            NUMERIC_EXPECTED_SPA,
            NUMERIC_EXPECTED_SPD,
            NUMERIC_EXPECTED_SPE,
        )
        row = [0.0] * width
        _encode_expected_stats(
            row, dex, base_species="Luvdisc", battle_species="Luvdisc", details="Luvdisc, M", belief=None
        )
        for slot in exp_slots:
            self.assertGreater(row[slot], 0.0)
        luv = dex.species_info("Luvdisc").base_stats
        self.assertAlmostEqual(
            row[NUMERIC_EXPECTED_SPE],
            min(1.0, _gen3_stat(luv["spe"], 100, ev=85, iv=31, hp=False) / _ACTUAL_STAT_DIVISOR),
        )

        # Sub-100 opponent expected-stats UNCHANGED: still the deterministic level-78 spread.
        row = [0.0] * width
        _encode_expected_stats(
            row, dex, base_species="Zapdos", battle_species="Zapdos", details="Zapdos, L78", belief=None
        )
        zap = dex.species_info("Zapdos").base_stats
        self.assertAlmostEqual(
            row[NUMERIC_EXPECTED_SPE],
            min(1.0, _gen3_stat(zap["spe"], 78, ev=85, iv=31, hp=False) / _ACTUAL_STAT_DIVISOR),
        )

    @unittest.skipUnless(_HAS_SHOWDOWN_CHECKOUT, "requires a real Gen 3 Showdown checkout for the dex + vocab")
    def test_l100_level_on_both_self_and_opponent_tokens_end_to_end(self) -> None:
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary

        dex = load_showdown_dex_cached(_SHOWDOWN_ROOT)
        vocab = gen3_category_vocabulary(_SHOWDOWN_ROOT)
        # Self (p1) active is an L100 Nosepass; opponent (p2) reveals an L100 Luvdisc. Both details
        # strings omit the level token (L100), so both tokens must encode NUMERIC_LEVEL=1.0.
        request = json.dumps(
            {
                "active": [{"moves": [{"move": "Rock Slide", "id": "rockslide"}], "trapped": False}],
                "side": {
                    "id": "p1",
                    "name": "Us",
                    "pokemon": [
                        {
                            "ident": "p1a: Nosepass",
                            "details": "Nosepass, F",
                            "condition": "100/100",
                            "active": True,
                            "moves": ["rockslide"],
                        },
                        {
                            "ident": "p1b: Snorlax",
                            "details": "Snorlax, L78",
                            "condition": "100/100",
                            "active": False,
                        },
                    ],
                },
            }
        )
        lines = [
            "|player|p1|Us|",
            "|player|p2|Them|",
            "|switch|p1a: Nosepass|Nosepass, F|100/100",
            "|switch|p2a: Luvdisc|Luvdisc|100/100",
            "|turn|1",
            "|request|" + request,
        ]
        replay = parse_showdown_replay(lines, battle_id="battle-gen3randombattle-1")
        state = normalize_for_player(replay, player_id="agent", configured_showdown_slot="p1")
        obs = observation_from_player_state(
            state, category_vocab=vocab, spec=V2_1_REPLAY_OBSERVATION_SPEC, dex=dex
        )
        self_active = FIELD_TOKEN_COUNT  # token 1
        opp_active = FIELD_TOKEN_COUNT + SELF_POKEMON_TOKEN_COUNT  # token 7
        self.assertEqual(obs.numeric_features[self_active][NUMERIC_LEVEL], 1.0)  # L100 self
        self.assertEqual(obs.numeric_features[opp_active][NUMERIC_LEVEL], 1.0)  # L100 opponent

    @unittest.skipUnless(_HAS_SHOWDOWN_CHECKOUT, "requires a real Gen 3 Showdown checkout for the dex")
    def test_unown_cosmetic_forme_resolves_to_base_types_and_stats(self) -> None:
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.showdown import (
            NUMERIC_BASE_ATK,
            NUMERIC_BASE_DEF,
            NUMERIC_BASE_HP,
            NUMERIC_BASE_SPA,
            NUMERIC_BASE_SPD,
            NUMERIC_BASE_SPE,
            CATEGORY_TYPE_2,
            _encode_pokemon_stats,
            _encode_species_type_categories,
            _species_info_base_fallback,
        )

        dex = load_showdown_dex_cached(_SHOWDOWN_ROOT)
        base = dex.species_info("unown")
        # The bug: the cosmetic forme is absent from the dex.
        self.assertIsNone(dex.species_info("Unown-C"))
        # The fix: base-species fallback resolves it to base Unown (Psychic, 48/72/48/72/48/48).
        resolved = _species_info_base_fallback(dex, "Unown-C")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.types, base.types)
        self.assertEqual(resolved.base_stats, base.base_stats)

        # Type categories: TYPE_1 = base Unown's (only) type; TYPE_2 stays pad (mono-type).
        cat = [""] * 8
        _encode_species_type_categories(cat, dex, "Unown-C")
        self.assertEqual(cat[CATEGORY_TYPE_1], f"type:{base.types[0]}")
        self.assertEqual(cat[CATEGORY_TYPE_2], "")

        # Base stats + level populated (Unown is always L100).
        num = [0.0] * 80
        _encode_pokemon_stats(num, dex, "Unown-C", "Unown-C")
        self.assertEqual(num[NUMERIC_LEVEL], 1.0)
        for slot, key in (
            (NUMERIC_BASE_HP, "hp"),
            (NUMERIC_BASE_ATK, "atk"),
            (NUMERIC_BASE_DEF, "def"),
            (NUMERIC_BASE_SPA, "spa"),
            (NUMERIC_BASE_SPD, "spd"),
            (NUMERIC_BASE_SPE, "spe"),
        ):
            self.assertAlmostEqual(num[slot], base.base_stats[key] / 200)

    @unittest.skipUnless(_HAS_SHOWDOWN_CHECKOUT, "requires a real Gen 3 Showdown checkout for the dex")
    def test_real_distinct_formes_are_not_collapsed_to_base(self) -> None:
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.showdown import _species_info_base_fallback

        dex = load_showdown_dex_cached(_SHOWDOWN_ROOT)
        # Real distinct Pokedex formes resolve on the DIRECT lookup, so the fallback returns them
        # unchanged and never collapses them to a base spread.
        for forme in ("Deoxys-Attack", "Deoxys-Defense", "Deoxys-Speed", "Castform"):
            direct = dex.species_info(forme)
            self.assertIsNotNone(direct, forme)
            self.assertEqual(_species_info_base_fallback(dex, forme).base_stats, direct.base_stats, forme)
        # Sanity: Deoxys-Attack's spread genuinely differs from base Deoxys (proves no collapse).
        self.assertNotEqual(
            dex.species_info("Deoxys-Attack").base_stats, dex.species_info("Deoxys").base_stats
        )


if __name__ == "__main__":
    unittest.main()
