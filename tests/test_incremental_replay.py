"""The incremental parser + persistent belief engine must match the batch reparse exactly.

LocalShowdownEnv feeds a persistent _ReplayParser and PublicBattleBeliefEngine one new line at a
time, then normalizes with the persistent engine. That path must be byte-identical to the old
behavior (re-parse the whole log + rebuild the belief engine from scratch every call) — these
tests pin that equivalence at every prefix of a battle so any divergence is caught immediately.
"""
import unittest
from pathlib import Path

from pokezero.belief import PublicBattleBeliefEngine
from pokezero.showdown import _ReplayParser, normalize_for_player, parse_showdown_replay

FIXTURE = Path(__file__).parent / "fixtures" / "showdown" / "p2_seat_replay.txt"

# A synthetic Gen 3 battle log exercising the parser/belief paths the thin fixture doesn't:
# switches, weather, side conditions, status, boosts, toxic escalation, baton pass, faint, win.
SYNTH_LINES = [
    "|player|p1|Alice|1|",
    "|player|p2|Bob|2|",
    "|switch|p1a: Tyranitar|Tyranitar, L82, M|100/100",
    "|switch|p2a: Skarmory|Skarmory, L80, F|100/100",
    "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar",
    "|turn|1",
    "|move|p2a: Skarmory|Spikes|p1a: Tyranitar",
    "|-sidestart|p1: Alice|Spikes",
    "|move|p1a: Tyranitar|Rock Slide|p2a: Skarmory",
    "|-damage|p2a: Skarmory|60/100",
    "|turn|2",
    "|move|p2a: Skarmory|Toxic|p1a: Tyranitar",
    "|-status|p1a: Tyranitar|tox",
    "|switch|p2a: Blissey|Blissey, L80, F|100/100",
    "|turn|3",
    "|-damage|p1a: Tyranitar|94/100 tox|[from] psn",
    "|move|p1a: Tyranitar|Dragon Dance|p1a: Tyranitar",
    "|-boost|p1a: Tyranitar|atk|1",
    "|-boost|p1a: Tyranitar|spe|1",
    "|turn|4",
    "|move|p1a: Tyranitar|Baton Pass|p1a: Tyranitar",
    "|switch|p1a: Celebi|Celebi, L80|100/100|[from] Baton Pass",
    "|turn|5",
    "|move|p2a: Blissey|Seismic Toss|p1a: Celebi",
    "|-damage|p1a: Celebi|75/100",
    "|faint|p2a: Blissey",
    "|win|Alice",
]


def _fixture_lines() -> list[str]:
    return FIXTURE.read_text(encoding="utf-8").splitlines()


class IncrementalReplayEquivalenceTest(unittest.TestCase):
    def _check(self, lines: list[str], player: str) -> None:
        # Mimic the env: one persistent parser + engine, fed only new lines as the log grows.
        parser = _ReplayParser()
        engine = PublicBattleBeliefEngine()
        fed_lines = 0
        fed_events = 0
        for k in range(1, len(lines) + 1):
            parser.feed(lines[fed_lines:k])
            fed_lines = k
            events = parser.public_events
            for event in events[fed_events:]:
                engine.ingest_event(event)
            fed_events = len(events)

            # Parser snapshot must equal a from-scratch batch parse of the same prefix.
            self.assertEqual(
                parser.snapshot(), parse_showdown_replay(lines[:k]), f"parser mismatch at prefix {k}"
            )

            incremental = normalize_for_player(
                parser.snapshot(),
                player_id="bot",
                configured_showdown_slot=player,
                belief_engine=engine,
            )
            batch = normalize_for_player(
                parse_showdown_replay(lines[:k]),
                player_id="bot",
                configured_showdown_slot=player,
            )
            self.assertEqual(incremental, batch, f"state mismatch at prefix {k} (player {player})")

    def test_fixture_p2(self) -> None:
        self._check(_fixture_lines(), "p2")

    def test_fixture_p1(self) -> None:
        self._check(_fixture_lines(), "p1")

    def test_synthetic_p1(self) -> None:
        self._check(SYNTH_LINES, "p1")

    def test_synthetic_p2(self) -> None:
        self._check(SYNTH_LINES, "p2")

    def test_snapshot_unaffected_by_later_feed(self) -> None:
        # A snapshot must not alias the parser's mutable accumulators.
        parser = _ReplayParser()
        parser.feed(SYNTH_LINES[:6])
        early = parser.snapshot()
        early_events = early.public_events
        parser.feed(SYNTH_LINES[6:])
        self.assertEqual(early.public_events, early_events)
        self.assertEqual(early, parse_showdown_replay(SYNTH_LINES[:6]))


if __name__ == "__main__":
    unittest.main()
