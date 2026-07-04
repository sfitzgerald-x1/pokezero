"""Repo-reproducible WS-1 C gate: replay the committed 5-capture corpus through full
observation building at spec v2, both perspectives, asserting shape/mask invariants and
ZERO category-vocab OOV — including a synthetic broken-Focus-Punch |cant| line (the
reachable emitter the original corpus happened not to contain).

Requires a built Gen 3 Showdown checkout (same skip gate as test_input_coverage); the
capture logs are controlled foul-play games committed under tests/fixtures.
"""

import os
import unittest
from pathlib import Path

SHOWDOWN_ROOT = Path(
    os.environ.get("POKEZERO_SHOWDOWN_ROOT", "/Users/scott/workspace/pokerena/vendor/pokemon-showdown")
)
CAPTURE_ROOT = Path(__file__).parent / "fixtures" / "showdown" / "capture"

# Emits the transition-token action id "focuspunch" (data/moves.ts onMoveAborted; Focus
# Punch is in the gen3 randbats movepools). Appended synthetically to one game because the
# 5-game corpus never contained a broken focus — exactly the coverage gap MED-3 flagged.
_SYNTHETIC_FOCUS_PUNCH = [
    "|cant|p2a: Azumarill|Focus Punch|Focus Punch",
    "|turn|99",
]


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class CorpusReplayGateTest(unittest.TestCase):
    def test_corpus_replays_through_spec_v2_with_invariants_and_zero_oov(self) -> None:
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.observation import ObservationFeatureMasks
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary
        from pokezero.showdown import (
            DEFAULT_REPLAY_OBSERVATION_SPEC,
            STATS_TOKEN_OFFSET,
            TRANSITION_TOKEN_OFFSET,
            _ReplayParser,
            normalize_for_player,
            observation_from_player_state,
        )

        spec = DEFAULT_REPLAY_OBSERVATION_SPEC
        dex = load_showdown_dex_cached(SHOWDOWN_ROOT)
        vocab = gen3_category_vocabulary(SHOWDOWN_ROOT)
        set_source = load_gen3_randbat_source_cached(SHOWDOWN_ROOT)
        k32 = ObservationFeatureMasks(transition_token_budget=32)

        capture_paths = sorted(CAPTURE_ROOT.glob("lines-*.log"))
        self.assertEqual(len(capture_paths), 5, capture_paths)

        boundaries = 0
        observations = 0
        for corpus_index, path in enumerate(capture_paths):
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
            if corpus_index == 0:
                lines = lines + _SYNTHETIC_FOCUS_PUNCH
            engines = {
                slot: PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=set_source)
                for slot in ("p1", "p2")
            }
            parser = _ReplayParser(battle_id=path.stem)
            fed_events = 0
            for line in lines:
                parser.feed([line])
                events = parser.public_events
                for event in events[fed_events:]:
                    for engine in engines.values():
                        engine.ingest_event(event)
                fed_events = len(events)
                if not (line.startswith("|turn|") or line.startswith("|win|")):
                    continue
                boundaries += 1
                replay = parser.snapshot()
                for slot in ("p1", "p2"):
                    state = normalize_for_player(
                        replay,
                        player_id=slot,
                        configured_showdown_slot=slot,
                        format_id="gen3randombattle",
                        belief_engine=engines[slot],
                    )
                    for masks in (None, k32):
                        kwargs = {"feature_masks": masks} if masks is not None else {}
                        observation = observation_from_player_state(
                            state, category_vocab=vocab, spec=spec, dex=dex, **kwargs
                        )
                        observation.validate(spec)
                        observations += 1
                        budget = masks.transition_token_budget if masks is not None else 128
                        filled = min(len(state.transition_tokens), budget)
                        tail = observation.attention_mask[TRANSITION_TOKEN_OFFSET:]
                        self.assertEqual(
                            list(tail), [index < filled for index in range(len(tail))]
                        )
                        for row_index in range(
                            TRANSITION_TOKEN_OFFSET + filled, spec.token_count
                        ):
                            self.assertEqual(
                                set(observation.numeric_features[row_index]), {0.0}
                            )
                            self.assertEqual(set(observation.categorical_ids[row_index]), {0})
                        self.assertEqual(
                            observation.attention_mask[STATS_TOKEN_OFFSET],
                            state.tendency_stats is not None,
                        )
                        for row_index in range(
                            TRANSITION_TOKEN_OFFSET, TRANSITION_TOKEN_OFFSET + filled
                        ):
                            row = observation.numeric_features[row_index]
                            # Tier-1 replay: all four materialized Tier-2 slots stay 0
                            # (residual, validity, CB bit; 120 is the always-zero
                            # investment reserve in every path).
                            self.assertEqual(row[117], 0.0)  # Tier-2 residual
                            self.assertEqual(row[118], 0.0)  # Tier-2 validity
                            self.assertEqual(row[119], 0.0)  # Tier-2 CB bit
                            self.assertEqual(row[120], 0.0)  # investment reserve

        self.assertGreater(boundaries, 100)
        self.assertGreater(observations, 500)
        # The zero-OOV invariant, now including the synthetic broken Focus Punch.
        self.assertEqual(vocab.observed_oov_tokens, frozenset())


if __name__ == "__main__":
    unittest.main()
