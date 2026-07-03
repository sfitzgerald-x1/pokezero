from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from pokezero.foulplay_bridge import (
    ControlledFoulPlayBenchmarkResult,
    ControlledFoulPlayConfig,
    ControlledFoulPlayGameResult,
    _ControlledBattleState,
    _choice_body_from_outgoing_message,
    _line_for_foulplay,
    _line_chunks_safe_for_foulplay,
    _requested_legal_action_masks_for_context,
    _is_terminal_protocol_line,
    _split_outgoing_showdown_message,
    _terminal_line_for_foulplay,
    _write_json,
)
from pokezero.env import TerminalState


class FoulPlayBridgeTest(unittest.TestCase):
    def test_split_outgoing_showdown_message_handles_room_and_global(self) -> None:
        self.assertEqual(
            _split_outgoing_showdown_message("battle-gen3randombattle-1|/choose move surf|7"),
            ("battle-gen3randombattle-1", "/choose move surf|7"),
        )
        self.assertEqual(
            _split_outgoing_showdown_message("|/trn FoulPlayBot,0,"),
            ("", "/trn FoulPlayBot,0,"),
        )

    def test_choice_body_from_foulplay_messages_normalizes_move_and_switch(self) -> None:
        self.assertEqual(_choice_body_from_outgoing_message("/choose move thunderbolt|12"), "move thunderbolt")
        self.assertEqual(_choice_body_from_outgoing_message("/choose move 3|12"), "move 3")
        self.assertEqual(_choice_body_from_outgoing_message("/switch 4|12"), "switch 4")
        self.assertIsNone(_choice_body_from_outgoing_message("/timer on"))

    def test_line_chunks_safe_for_foulplay_drops_noise_and_splits_sensitive_lines(self) -> None:
        self.assertEqual(
            _line_chunks_safe_for_foulplay(
                (
                    "|t:|1783052150",
                    "|",
                    "|gametype|singles",
                    "|player|p1|PokeZeroBot|",
                    "|start",
                    "|request|{}",
                )
            ),
            (
                ("|gametype|singles",),
                ("|player|p1|PokeZeroBot|",),
                ("|start",),
                ("|request|{}",),
            ),
        )

    def test_line_for_foulplay_injects_rqid_into_battlestream_request_copy(self) -> None:
        state = _ControlledBattleState(
            battle_id="battle-gen3randombattle-1",
            seed=1,
            format_id="gen3randombattle",
        )

        line = _line_for_foulplay(state, '|request|{"active":[{"moves":[]}],"side":{"id":"p2"}}')

        self.assertIn('"rqid":1', line)
        self.assertEqual(state.next_foulplay_rqid, 2)
        self.assertEqual(_line_for_foulplay(state, '|request|{"rqid":99}'), '|request|{"rqid":99}')

    def test_requested_legal_action_masks_can_hide_opponent_private_mask(self) -> None:
        class Observation:
            def __init__(self, mask: tuple[bool, ...]) -> None:
                self.legal_action_mask = mask

        observations = {
            "p1": Observation((True, False)),
            "p2": Observation((False, True)),
        }

        self.assertEqual(
            _requested_legal_action_masks_for_context(
                observations,  # type: ignore[arg-type]
                acting_player="p1",
                opponent_legal_mask_mode="hidden",
            ),
            {"p1": (True, False)},
        )
        self.assertEqual(
            _requested_legal_action_masks_for_context(
                observations,  # type: ignore[arg-type]
                acting_player="p1",
                opponent_legal_mask_mode="privileged",
            ),
            {"p1": (True, False), "p2": (False, True)},
        )

    def test_terminal_line_for_foulplay_uses_configured_display_names(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
        )

        self.assertEqual(
            _terminal_line_for_foulplay(TerminalState(winner="p1", turn_count=10), config),
            "|win|PokeZeroBot",
        )
        self.assertEqual(
            _terminal_line_for_foulplay(TerminalState(winner="p2", turn_count=10), config),
            "|win|FoulPlayBot",
        )
        self.assertEqual(
            _terminal_line_for_foulplay(TerminalState(winner=None, turn_count=250, capped=True), config),
            "|tie|",
        )

    def test_is_terminal_protocol_line_detects_win_and_tie(self) -> None:
        self.assertTrue(_is_terminal_protocol_line("|win|PokeZeroBot"))
        self.assertTrue(_is_terminal_protocol_line("|tie|"))
        self.assertFalse(_is_terminal_protocol_line("|turn|2"))

    def test_benchmark_payload_summarizes_root_puct_metrics(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint=Path("checkpoint.pt"),
            showdown_root=Path("/showdown"),
            games=2,
        )
        result = ControlledFoulPlayBenchmarkResult(
            config=config,
            policy_id="policy+root-puct",
            games=(
                ControlledFoulPlayGameResult(
                    battle_id="battle-1",
                    seed=1,
                    winner="PokeZeroBot",
                    pokezero_won=True,
                    decision_rounds=3,
                    pokezero_decisions=3,
                    root_puct_searches=3,
                    root_puct_fallbacks=0,
                    root_puct_average_elapsed_seconds=0.2,
                ),
                ControlledFoulPlayGameResult(
                    battle_id="battle-2",
                    seed=2,
                    winner="FoulPlayBot",
                    pokezero_won=False,
                    decision_rounds=4,
                    pokezero_decisions=4,
                    root_puct_searches=2,
                    root_puct_fallbacks=2,
                    root_puct_average_elapsed_seconds=0.4,
                ),
            ),
        )

        payload = result.to_dict()

        self.assertEqual(payload["schema_version"], "pokezero.controlled-foulplay-benchmark.v1")
        self.assertEqual(payload["wins"], 1)
        self.assertEqual(payload["completed_games"], 2)
        self.assertEqual(payload["win_rate"], 0.5)
        self.assertEqual(payload["root_puct"]["searches"], 5)
        self.assertEqual(payload["root_puct"]["fallbacks"], 2)
        self.assertEqual(payload["root_puct"]["opponent_legal_mask_mode"], "hidden")
        self.assertEqual(payload["root_puct"]["foulplay_search_time_ms"], 1000)
        self.assertAlmostEqual(payload["root_puct"]["average_elapsed_seconds"], 0.3)

    def test_write_json_creates_parent_directory_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "summary.json"

            _write_json(path, {"b": 2, "a": 1})

            self.assertEqual(path.read_text(), '{\n  "a": 1,\n  "b": 2\n}\n')


if __name__ == "__main__":
    unittest.main()
