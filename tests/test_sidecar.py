import unittest

from pokezero.randbat import Gen3RandbatSource
from pokezero.sidecar import BeliefSidecarState, _split_protocol_message, render_index_html


class SidecarWebviewTest(unittest.TestCase):
    def test_state_payload_exposes_belief_source_and_public_lines(self) -> None:
        set_source = Gen3RandbatSource.from_data(
            {
                "xatu": {
                    "level": 84,
                    "sets": [
                        {
                            "role": "Setup Sweeper",
                            "movepool": ["calmmind", "hiddenpowerfire", "psychic", "rest"],
                            "abilities": ["Early Bird"],
                        }
                    ],
                }
            }
        )
        state = BeliefSidecarState(
            room_id="battle-gen3randombattle-1",
            set_source=set_source,
            perspective="p2",
        )

        changed = state.ingest_lines(
            [
                "|player|p1|Friend|",
                "|player|p2|PokeZero|",
                "|switch|p1a: Xatu|Xatu, L84|100/100",
                "|request|{\"side\":{\"id\":\"p2\"}}",
                "|move|p1a: Xatu|Psychic|p2a: Charizard",
            ]
        )
        payload = state.payload()

        self.assertTrue(changed)
        self.assertEqual(payload["room_id"], "battle-gen3randombattle-1")
        self.assertEqual(payload["perspective"], "p2")
        self.assertIn("source_hash", payload["source"])
        self.assertNotIn("|request|", "\n".join(payload["recent_public_lines"]))
        self.assertEqual(payload["player_view"]["opponent_pokemon"][0]["species"], "Xatu")
        self.assertEqual(payload["player_view"]["opponent_pokemon"][0]["revealed_moves"], ["Psychic"])

    def test_html_contains_sidecar_view_contract(self) -> None:
        html = render_index_html()

        self.assertIn("PokeZero Belief Sidecar", html)
        self.assertIn("/api/state", html)
        self.assertIn("new EventSource('/api/events')", html)
        self.assertIn("Surviving variants", html)

    def test_protocol_split_handles_global_and_room_chunks(self) -> None:
        chunks = _split_protocol_message("|challstr|abc|def\n>battle-gen3randombattle-1\n|turn|1\n|win|Friend")

        self.assertEqual(chunks[0], (None, ["|challstr|abc|def"]))
        self.assertEqual(chunks[1], ("battle-gen3randombattle-1", ["|turn|1", "|win|Friend"]))


if __name__ == "__main__":
    unittest.main()
