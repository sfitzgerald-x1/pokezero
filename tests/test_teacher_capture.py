"""Unit tests for decoding a teacher's Showdown /choose string into our action index."""

from types import SimpleNamespace

from pokezero.teacher_capture import action_index_from_choice_string


def _mon(species, active=False):
    return SimpleNamespace(species=species, active=active)


def _state(mask, moves):
    return SimpleNamespace(
        self_team=[_mon("Skarmory", active=True), _mon("Blissey"), _mon("Starmie")],
        legal_action_mask=tuple(mask),
        request={"active": [{"moves": [{"move": m, "id": m.lower()} for m in moves]}]},
    )


MOVES = ["Spikes", "Roar", "Toxic", "Whirlwind"]
# 3-mon team, active = Skarmory (team 0): moves are actions 0-3; switch targets are
# Blissey->action 4, Starmie->action 5; actions 6-8 are unused (always illegal).
#          m0    m1     m2    m3     sw:Blissey  sw:Starmie  6      7      8
LEGAL = (True, True, False, True,   False,      True,       False, False, False)


def test_move_by_slot_legal():
    assert action_index_from_choice_string(_state(LEGAL, MOVES), "move 1") == 0
    assert action_index_from_choice_string(_state(LEGAL, MOVES), "move 4") == 3


def test_move_by_slot_illegal_returns_none():
    # slot 2 (move 3 -> index 2) is masked illegal (e.g. disabled)
    assert action_index_from_choice_string(_state(LEGAL, MOVES), "move 3") is None


def test_move_by_name():
    assert action_index_from_choice_string(_state(LEGAL, MOVES), "move roar") == 1
    assert action_index_from_choice_string(_state(LEGAL, MOVES), "move spikes") == 0


def test_switch_by_slot():
    # active = team 0 (Skarmory); switch targets in order are team 1 (Blissey)->a4, team 2 (Starmie)->a5.
    # a4 is illegal, a5 legal, so 'switch 3' (team index 2 = Starmie) -> action 5.
    assert action_index_from_choice_string(_state(LEGAL, MOVES), "switch 3") == 5
    # 'switch 2' (Blissey -> a4) is masked illegal.
    assert action_index_from_choice_string(_state(LEGAL, MOVES), "switch 2") is None


def test_switch_by_species():
    assert action_index_from_choice_string(_state(LEGAL, MOVES), "switch Starmie") == 5
    assert action_index_from_choice_string(_state(LEGAL, MOVES), "switch starmie") == 5


def test_leading_choose_token_tolerated():
    assert action_index_from_choice_string(_state(LEGAL, MOVES), "/choose move 1") == 0


def test_move_with_target_and_gimmick_suffix():
    # e.g. doubles target or 'terastallize' trailing token — slot still parses from the first arg.
    assert action_index_from_choice_string(_state(LEGAL, MOVES), "move 4 terastallize") == 3


def test_hidden_power_matched_by_display_name():
    # foul-play submits the typed Hidden Power id while the request id is just "hiddenpower".
    state = SimpleNamespace(
        self_team=[_mon("Zapdos", active=True), _mon("Blissey")],
        legal_action_mask=(True, True, False, False, False, True, False, False, False),
        request={"active": [{"moves": [
            {"move": "Thunderbolt", "id": "thunderbolt"},
            {"move": "Hidden Power Ice 70", "id": "hiddenpower"},
        ]}]},
    )
    assert action_index_from_choice_string(state, "move hiddenpowerice70") == 1
    assert action_index_from_choice_string(state, "move hiddenpower") == 1  # id form still works


def test_non_battle_choices_return_none():
    for choice in ("default", "pass", "team 123456", "", "move", "switch"):
        assert action_index_from_choice_string(_state(LEGAL, MOVES), choice) is None
