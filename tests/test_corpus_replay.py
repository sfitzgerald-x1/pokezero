"""Repo-reproducible WS-1 C gate: replay the committed 5-capture corpus through full
observation building in BOTH supported schema modes (spec v2 AND spec v2.1), both
perspectives, asserting shape/mask invariants, the v2-prefix property of the dual encoder,
and ZERO category-vocab OOV — including a synthetic broken-Focus-Punch |cant| line (the
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

# The Hitmonlee game: p2's Hitmonlee reveals exactly Earthquake, Substitute, and Reversal
# before fainting — the committed fixture for the v2.1 PP-validity bucket-alignment gate.
_HITMONLEE_GAME = "lines-battle-gen3randombattle-controlled-20260710002.log"
_HITMONLEE_REVEALED = {"earthquake", "substitute", "reversal"}


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class CorpusReplayGateTest(unittest.TestCase):
    def test_corpus_replays_through_both_schemas_with_invariants_and_zero_oov(self) -> None:
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.observation import ObservationFeatureMasks
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary
        from pokezero.showdown import (
            CATEGORY_BELIEF_MOVE_OFFSET,
            CATEGORY_MOVE_PRIORITY,
            NUMERIC_OPP_MOVE_PP_OFFSET,
            NUMERIC_OPP_MOVE_PP_VALID_OFFSET,
            NUMERIC_REVEALED_MOVE_COUNT,
            NUMERIC_TIER2_CB_PINNED,
            NUMERIC_TIER2_INVESTMENT_PINNED,
            OPPONENT_POKEMON_TOKEN_OFFSET,
            OPPONENT_POKEMON_TOKEN_COUNT,
            STATS_TOKEN_OFFSET,
            TRANSITION_TOKEN_OFFSET,
            V2_1_REPLAY_OBSERVATION_SPEC,
            V2_REPLAY_OBSERVATION_SPEC,
            _ReplayParser,
            _normalize_identifier,
            normalize_for_player,
            observation_from_player_state,
        )
        from pokezero.transitions import TOKEN_KIND_MOVE

        dex = load_showdown_dex_cached(SHOWDOWN_ROOT)
        vocab = gen3_category_vocabulary(SHOWDOWN_ROOT)
        set_source = load_gen3_randbat_source_cached(SHOWDOWN_ROOT)
        k32 = ObservationFeatureMasks(transition_token_budget=32)
        v2_width = V2_REPLAY_OBSERVATION_SPEC.numeric_feature_count

        capture_paths = sorted(CAPTURE_ROOT.glob("lines-*.log"))
        self.assertEqual(len(capture_paths), 5, capture_paths)

        boundaries = 0
        observations = 0
        defender_slot_rows = 0
        validity_bits_seen = 0
        hitmonlee_gate_checked = False
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
                        encoded = {}
                        for spec in (V2_REPLAY_OBSERVATION_SPEC, V2_1_REPLAY_OBSERVATION_SPEC):
                            observation = observation_from_player_state(
                                state, category_vocab=vocab, spec=spec, dex=dex, **kwargs
                            )
                            observation.validate(spec)
                            encoded[spec.schema_version] = observation
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
                                # investment reserve in every path — carried forward
                                # unchanged into the v2.1 census).
                                self.assertEqual(row[117], 0.0)  # Tier-2 residual
                                self.assertEqual(row[118], 0.0)  # Tier-2 validity
                                self.assertEqual(row[119], 0.0)  # Tier-2 CB bit
                                self.assertEqual(row[120], 0.0)  # investment reserve

                        v2 = encoded["pokezero.observation.v2"]
                        v2_1 = encoded["pokezero.observation.v2.1"]
                        # Dual-encoder prefix property: the v2 encode is a byte prefix of the
                        # v2.1 numeric rows, and the categoricals agree everywhere except the
                        # defender slot on transition rows.
                        for v2_row, v21_row in zip(v2.numeric_features, v2_1.numeric_features):
                            self.assertEqual(tuple(v2_row), tuple(v21_row[:v2_width]))
                        for row_index, (v2_row, v21_row) in enumerate(
                            zip(v2.categorical_ids, v2_1.categorical_ids)
                        ):
                            for column, (a, b) in enumerate(zip(v2_row, v21_row)):
                                if (
                                    column == CATEGORY_MOVE_PRIORITY
                                    and row_index >= TRANSITION_TOKEN_OFFSET
                                ):
                                    if b != 0:
                                        defender_slot_rows += 1
                                    # v2 must keep the slot dark on transition rows.
                                    self.assertEqual(a, 0)
                                    continue
                                self.assertEqual(a, b)

                        # v2.1 PP-validity invariant on every opponent row: the bit count
                        # equals the revealed-move count (revealed moves are never evicted
                        # from the buckets), and bits only sit on occupied buckets.
                        if masks is None:
                            for mon_index in range(OPPONENT_POKEMON_TOKEN_COUNT):
                                row = v2_1.numeric_features[
                                    OPPONENT_POKEMON_TOKEN_OFFSET + mon_index
                                ]
                                cat_row = v2_1.categorical_ids[
                                    OPPONENT_POKEMON_TOKEN_OFFSET + mon_index
                                ]
                                bits = [
                                    column
                                    for column in range(16)
                                    if row[NUMERIC_OPP_MOVE_PP_VALID_OFFSET + column] == 1.0
                                ]
                                self.assertEqual(
                                    len(bits), int(row[NUMERIC_REVEALED_MOVE_COUNT])
                                )
                                validity_bits_seen += len(bits)
                                for column in bits:
                                    self.assertNotEqual(
                                        cat_row[CATEGORY_BELIEF_MOVE_OFFSET + column], 0
                                    )
                                    # A PP fraction may legitimately be 0.0 (the closed
                                    # collision) but never appears on an INVALID bucket.
                                for column in range(16):
                                    if row[NUMERIC_OPP_MOVE_PP_OFFSET + column] > 0.0:
                                        self.assertIn(column, bits)
                                # Tier-1 replay (no tier2 tracker/inference ran): both
                                # per-mon pinned Tier-2 surfaces stay dark, mirroring the
                                # tt-row 117..120 invariant above.
                                self.assertEqual(row[NUMERIC_TIER2_CB_PINNED], 0.0)
                                self.assertEqual(row[NUMERIC_TIER2_INVESTMENT_PINNED], 0.0)

            # The committed Hitmonlee fixture gate, at the game's FINAL boundary (the
            # captures end at their last |turn|; Hitmonlee fainted long before, so its
            # reveal set is frozen): exactly 3 validity bits — Earthquake, Substitute,
            # Reversal — bucket-aligned with the PP columns, from p1's perspective.
            if path.name == _HITMONLEE_GAME:
                final_state = normalize_for_player(
                    parser.snapshot(),
                    player_id="p1",
                    configured_showdown_slot="p1",
                    format_id="gen3randombattle",
                    belief_engine=engines["p1"],
                )
                final_obs = observation_from_player_state(
                    final_state, category_vocab=vocab, spec=V2_1_REPLAY_OBSERVATION_SPEC, dex=dex
                )
                mon_index = next(
                    index
                    for index, mon in enumerate(final_state.opponent_team)
                    if _normalize_identifier(mon.species) == "hitmonlee"
                )
                row = final_obs.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET + mon_index]
                cat_row = final_obs.categorical_ids[OPPONENT_POKEMON_TOKEN_OFFSET + mon_index]
                bits = [
                    column
                    for column in range(16)
                    if row[NUMERIC_OPP_MOVE_PP_VALID_OFFSET + column] == 1.0
                ]
                self.assertEqual(len(bits), 3, bits)
                revealed_rows = {
                    vocab.encode(f"belief:possible_move:{move}") for move in _HITMONLEE_REVEALED
                }
                self.assertEqual(
                    {cat_row[CATEGORY_BELIEF_MOVE_OFFSET + column] for column in bits},
                    revealed_rows,
                )
                # Bucket alignment with the PP columns: every revealed bucket carries a PP
                # fraction (none were ledgered to 0 in this game).
                for column in bits:
                    self.assertGreater(row[NUMERIC_OPP_MOVE_PP_OFFSET + column], 0.0)
                hitmonlee_gate_checked = True

            # Defender identity fires on real fixture games: after the full game, the last
            # v2.1 encode's transition rows carried defender species on move tokens.
            move_tokens = [
                token for token in state.transition_tokens if token.kind == TOKEN_KIND_MOVE
            ]
            if move_tokens:
                self.assertTrue(
                    all(token.defender_species for token in move_tokens),
                    f"{path.name}: move transition tokens missing defender species",
                )

        self.assertGreater(boundaries, 100)
        self.assertGreater(observations, 1000)  # two schemas x two masks x two slots
        self.assertGreater(defender_slot_rows, 100)
        self.assertGreater(validity_bits_seen, 100)
        self.assertTrue(hitmonlee_gate_checked, "Hitmonlee validity-bit fixture gate never ran")
        # The zero-OOV invariant, now including the synthetic broken Focus Punch and the
        # v2.1 defender species labels.
        self.assertEqual(vocab.observed_oov_tokens, frozenset())


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class TurnMergedCorpusReplayGateTest(unittest.TestCase):
    """The corpus gate in schema v2.2 (TURN-MERGED transition tokens).

    Same 5-game corpus, both perspectives, every boundary — through the v2.2 spec with
    the turn-merged vocabulary, asserting shape/mask invariants and ZERO OOV for the
    tt_phase/tt2_* families. K BUDGET UNIT NOTE: the budget=16 variant here keeps 16
    TURN tokens (~16 turns), not 16 actions — the unit changed with the schema (see the
    design doc's turn-merged addendum)."""

    def test_corpus_replays_through_v2_2_with_zero_oov(self) -> None:
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.observation import ObservationFeatureMasks
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary
        from pokezero.showdown import (
            NUMERIC_TM2_PRESENT,
            TRANSITION_TOKEN_OFFSET,
            V2_2_REPLAY_OBSERVATION_SPEC,
            _ReplayParser,
            normalize_for_player,
            observation_from_player_state,
        )
        from pokezero.turn_merged import SUB_BLOCK_ACTION

        spec = V2_2_REPLAY_OBSERVATION_SPEC
        dex = load_showdown_dex_cached(SHOWDOWN_ROOT)
        vocab = gen3_category_vocabulary(SHOWDOWN_ROOT, include_turn_merged=True)
        set_source = load_gen3_randbat_source_cached(SHOWDOWN_ROOT)
        mask_variants = (
            ObservationFeatureMasks(),
            ObservationFeatureMasks(transition_token_budget=16),
        )

        capture_paths = sorted(CAPTURE_ROOT.glob("lines-*.log"))
        self.assertEqual(len(capture_paths), 5, capture_paths)

        observations = 0
        merged_seen = 0
        second_action_rows = 0
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
                replay = parser.snapshot()
                for slot in ("p1", "p2"):
                    state = normalize_for_player(
                        replay,
                        player_id=slot,
                        configured_showdown_slot=slot,
                        format_id="gen3randombattle",
                        belief_engine=engines[slot],
                        include_turn_merged=True,
                    )
                    merged_seen = max(merged_seen, len(state.turn_merged_tokens))
                    second_action_rows += sum(
                        1
                        for token in state.turn_merged_tokens
                        if token.second.status == SUB_BLOCK_ACTION
                    )
                    for masks in mask_variants:
                        observation = observation_from_player_state(
                            state, category_vocab=vocab, spec=spec, dex=dex, feature_masks=masks
                        )
                        observation.validate(spec)
                        observations += 1
                        filled = min(
                            len(state.turn_merged_tokens), masks.transition_token_budget
                        )
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
                        # Tier-1 replay: sub-block Tier-2 columns stay dark in both halves.
                        for row_index in range(
                            TRANSITION_TOKEN_OFFSET, TRANSITION_TOKEN_OFFSET + filled
                        ):
                            row = observation.numeric_features[row_index]
                            self.assertEqual(row[117], 0.0)  # first residual
                            self.assertEqual(row[118], 0.0)  # first validity
                            self.assertEqual(row[119], 0.0)  # first CB bit
                            self.assertEqual(row[120], 0.0)  # investment reserve
                        # NUMERIC_TM2_PRESENT fires exactly on executed second halves.
                        visible = state.turn_merged_tokens[-filled:] if filled else ()
                        for offset, token in enumerate(visible):
                            self.assertEqual(
                                observation.numeric_features[TRANSITION_TOKEN_OFFSET + offset][
                                    NUMERIC_TM2_PRESENT
                                ],
                                1.0 if token.second.status == SUB_BLOCK_ACTION else 0.0,
                            )

        self.assertGreater(observations, 500)
        self.assertGreater(merged_seen, 20)
        self.assertGreater(second_action_rows, 100)
        # Zero OOV across the merged families (tt_phase / tt2_*), both perspectives.
        self.assertEqual(vocab.observed_oov_tokens, frozenset())


if __name__ == "__main__":
    unittest.main()
