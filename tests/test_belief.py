from pathlib import Path
import unittest

from pokezero.belief import CandidateSetSummary, PublicBattleBeliefEngine
from pokezero.showdown import parse_showdown_replay


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "showdown"


def fixture_lines(name: str) -> list[str]:
    return (FIXTURE_ROOT / name).read_text(encoding="utf-8").splitlines()


class FakeSetSource:
    def summarize(self, *, format_id, species, revealed_moves):
        candidate_count = max(1, 4 - len(revealed_moves))
        uncertainty = candidate_count / 4.0
        return CandidateSetSummary(
            species=species,
            candidate_count=candidate_count,
            uncertainty=uncertainty,
        )


class PublicBattleBeliefEngineTest(unittest.TestCase):
    def test_tracks_public_reveals_moves_and_conditions(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        snapshot = PublicBattleBeliefEngine.from_events(replay.public_events).snapshot()
        p1_species = [pokemon.species for pokemon in snapshot.side("p1")]
        xatu = next(pokemon for pokemon in snapshot.side("p1") if pokemon.species == "Xatu")

        self.assertEqual(p1_species, ["Arcanine", "Xatu"])
        self.assertFalse(snapshot.side("p1")[0].active)
        self.assertTrue(xatu.active)
        self.assertEqual(xatu.condition, "70/100")
        self.assertEqual(xatu.revealed_moves, ("Psychic",))
        self.assertNotIn("Thunder Wave", xatu.revealed_moves)

    def test_player_view_is_overlay_ready_and_player_relative(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        view = PublicBattleBeliefEngine.from_events(replay.public_events).snapshot().for_player("p2")
        payload = view.to_overlay_payload()

        self.assertEqual(payload["self_slot"], "p2")
        self.assertEqual(payload["opponent_slot"], "p1")
        self.assertEqual(
            [pokemon["species"] for pokemon in payload["opponent_pokemon"]],
            ["Arcanine", "Xatu"],
        )
        self.assertEqual(payload["opponent_pokemon"][1]["revealed_moves"], ["Psychic"])

    def test_set_source_can_attach_candidate_summary_without_engine_coupling(self) -> None:
        replay = parse_showdown_replay(fixture_lines("p2_seat_replay.txt"), battle_id="battle-gen3randombattle-1")

        snapshot = PublicBattleBeliefEngine.from_events(
            replay.public_events,
            format_id="gen3randombattle",
            set_source=FakeSetSource(),
        ).snapshot()
        xatu = next(pokemon for pokemon in snapshot.side("p1") if pokemon.species == "Xatu")

        self.assertEqual(xatu.candidate_set_count, 3)
        self.assertEqual(xatu.uncertainty, 0.75)


if __name__ == "__main__":
    unittest.main()
