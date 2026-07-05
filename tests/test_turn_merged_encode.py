"""Column-level tests for the v2.2 (turn-merged) transition encode.

v2.2 is the third entry in the checkpoint-driven dual-schema table (#512): every v2.1
block carries forward; only the transition surface changes to turn-merged tokens. These
tests pin the column CONTENT contract through the named constants, the spec-table wiring,
and the K-truncation semantics — including the #514 pinned-bit-survives-truncation
behavior, which matters MORE here because the budget unit changed from actions to turns.
"""

import os
import unittest
from dataclasses import replace as dc_replace
from pathlib import Path

from pokezero.observation import (
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION_V2_1,
    OBSERVATION_SCHEMA_VERSION_V2_2,
    SUPPORTED_OBSERVATION_SCHEMA_VERSIONS,
    ObservationFeatureMasks,
)
from pokezero.showdown import (
    CATEGORY_MOVE_PRIORITY,
    CATEGORY_PRIMARY,
    CATEGORY_SECONDARY,
    CATEGORY_SLOT,
    CATEGORY_TM_FIRST_CANT,
    CATEGORY_TM_FIRST_KIND,
    CATEGORY_TM_SECOND_ACTION,
    CATEGORY_TM_SECOND_DEFENDER,
    CATEGORY_TM_SECOND_KIND,
    CATEGORY_TM_SECOND_SPECIES,
    NUMERIC_TIER2_CB_PINNED,
    NUMERIC_TIER2_INVESTMENT_PINNED,
    NUMERIC_TM2_CB_BIT,
    NUMERIC_TM2_INVESTMENT,
    NUMERIC_TM2_DAMAGE_FRACTION,
    NUMERIC_TM2_PRESENT,
    NUMERIC_TT_CALLED,
    NUMERIC_TT_CB_BIT,
    NUMERIC_TT_DAMAGE_FRACTION,
    NUMERIC_TT_INVESTMENT_BIT,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    REPLAY_OBSERVATION_SPECS_BY_SCHEMA,
    TRANSITION_TOKEN_OFFSET,
    V2_1_REPLAY_OBSERVATION_SPEC,
    V2_2_REPLAY_OBSERVATION_SPEC,
    _TM_SUB_BLOCK_ACTION,
    normalize_for_player,
    observation_from_player_state,
    observation_spec_for_schema,
    parse_showdown_replay,
)
from pokezero.transitions import TOKEN_KIND_MOVE
from pokezero.turn_merged import (
    PHASE_EXTRA,
    PHASE_LEAD,
    PHASE_REPLACEMENT,
    PHASE_TURN,
    SUB_BLOCK_ABSENT,
    SUB_BLOCK_ACTION,
    SUB_BLOCK_NEGATED,
    annotate_turn_merged_tokens,
)

SHOWDOWN_ROOT = Path(
    os.environ.get("POKEZERO_SHOWDOWN_ROOT", "/Users/scott/workspace/pokerena/vendor/pokemon-showdown")
)


class SchemaTableTest(unittest.TestCase):
    def test_v2_2_is_a_supported_schema_entry_but_not_the_default(self) -> None:
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V2_2, SUPPORTED_OBSERVATION_SCHEMA_VERSIONS)
        self.assertIn(OBSERVATION_SCHEMA_VERSION_V2_2, REPLAY_OBSERVATION_SPECS_BY_SCHEMA)
        self.assertIs(
            observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION_V2_2),
            V2_2_REPLAY_OBSERVATION_SPEC,
        )
        # Turn-merged is the batch-3 ablation arm, not the new default.
        self.assertEqual(OBSERVATION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION_V2_1)

    def test_v2_2_widths_extend_the_v2_1_census(self) -> None:
        self.assertEqual(
            V2_2_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
            V2_1_REPLAY_OBSERVATION_SPEC.numeric_feature_count + 13,
        )
        self.assertEqual(
            V2_2_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
            V2_1_REPLAY_OBSERVATION_SPEC.categorical_feature_count + 12,
        )
        self.assertEqual(
            V2_2_REPLAY_OBSERVATION_SPEC.token_count, V2_1_REPLAY_OBSERVATION_SPEC.token_count
        )

    def test_showdown_literal_matches_turn_merged_constant(self) -> None:
        self.assertEqual(_TM_SUB_BLOCK_ACTION, SUB_BLOCK_ACTION)

    def test_vocab_families_match_turn_merged_constants(self) -> None:
        from pokezero.randbat_vocab import TURN_MERGED_PHASES, TURN_MERGED_SECOND_STATUSES

        self.assertEqual(
            set(TURN_MERGED_PHASES), {PHASE_TURN, PHASE_LEAD, PHASE_REPLACEMENT, PHASE_EXTRA}
        )
        from pokezero.turn_merged import SUB_BLOCK_PENDING

        self.assertEqual(
            set(TURN_MERGED_SECOND_STATUSES),
            {SUB_BLOCK_NEGATED, SUB_BLOCK_PENDING, SUB_BLOCK_ABSENT},
        )


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class TurnMergedEncodeTest(unittest.TestCase):
    @staticmethod
    def _vocab():
        from pokezero.randbat_vocab import gen3_category_vocabulary

        return gen3_category_vocabulary(SHOWDOWN_ROOT, include_turn_merged=True)

    def _state(self, lines):
        replay = parse_showdown_replay(lines, battle_id="turn-merged-encode")
        return normalize_for_player(
            replay,
            player_id="p1",
            configured_showdown_slot="p1",
            format_id="gen3randombattle",
            include_turn_merged=True,
        )

    def _encode(self, state, budget=128, **kwargs):
        observation = observation_from_player_state(
            state,
            category_vocab=self._vocab(),
            spec=V2_2_REPLAY_OBSERVATION_SPEC,
            feature_masks=ObservationFeatureMasks(transition_token_budget=budget, **kwargs),
        )
        observation.validate(V2_2_REPLAY_OBSERVATION_SPEC)
        return observation

    def _lines(self):
        return [
            "|player|p1|Alice|",
            "|player|p2|Bob|",
            "|switch|p1a: Snorlax|Snorlax, L80|100/100",
            "|switch|p2a: Skarmory|Skarmory, L76|100/100",
            "|turn|1",
            "|cant|p1a: Snorlax|slp",
            "|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax",
            "|move|p1a: Snorlax|Body Slam|p2a: Skarmory|[from] Sleep Talk",
            "|-damage|p2a: Skarmory|70/100",
            "|move|p2a: Skarmory|Drill Peck|p1a: Snorlax",
            "|-damage|p1a: Snorlax|85/100",
            "|upkeep",
            "|turn|2",
        ]

    def test_merged_row_columns_and_mask(self) -> None:
        state = self._state(self._lines())
        observation = self._encode(state)
        vocab = self._vocab()
        self.assertEqual(len(state.turn_merged_tokens), 2)  # lead pair + turn 1
        lead_row = TRANSITION_TOKEN_OFFSET
        turn_row = TRANSITION_TOKEN_OFFSET + 1
        self.assertTrue(observation.attention_mask[lead_row])
        self.assertTrue(observation.attention_mask[turn_row])
        self.assertFalse(observation.attention_mask[turn_row + 1])

        def cat(row, column):
            return observation.categorical_ids[row][column]

        # Lead pair: SLOT carries the phase; second sub-block is Skarmory's send-out.
        self.assertEqual(cat(lead_row, CATEGORY_SLOT), vocab.encode("tt_phase:lead"))
        self.assertEqual(cat(lead_row, CATEGORY_TM_SECOND_KIND), vocab.encode("tt2_kind:switch"))
        self.assertEqual(
            cat(lead_row, CATEGORY_TM_SECOND_ACTION), vocab.encode("tt2_species:Skarmory")
        )
        # RestTalk turn: first sub-block is the called Body Slam with the cant collapse;
        # tt_kind moves to its appended column; MOVE_PRIORITY carries the DEFENDER
        # (exactly the v2.1 per-action semantics).
        self.assertEqual(cat(turn_row, CATEGORY_SLOT), vocab.encode("tt_phase:turn"))
        self.assertEqual(cat(turn_row, CATEGORY_PRIMARY), vocab.encode("species:Snorlax"))
        self.assertEqual(cat(turn_row, CATEGORY_SECONDARY), vocab.encode("move:bodyslam"))
        self.assertEqual(cat(turn_row, CATEGORY_TM_FIRST_KIND), vocab.encode("tt_kind:move"))
        self.assertEqual(cat(turn_row, CATEGORY_MOVE_PRIORITY), vocab.encode("species:Skarmory"))
        self.assertEqual(cat(turn_row, CATEGORY_TM_FIRST_CANT), vocab.encode("cant:slp"))
        # Second sub-block: Skarmory's Drill Peck on the tt2_ columns, defender bound.
        self.assertEqual(cat(turn_row, CATEGORY_TM_SECOND_KIND), vocab.encode("tt2_kind:move"))
        self.assertEqual(
            cat(turn_row, CATEGORY_TM_SECOND_SPECIES), vocab.encode("tt2_species:Skarmory")
        )
        self.assertEqual(
            cat(turn_row, CATEGORY_TM_SECOND_ACTION), vocab.encode("tt2_move:drillpeck")
        )
        self.assertEqual(
            cat(turn_row, CATEGORY_TM_SECOND_DEFENDER), vocab.encode("tt2_species:Snorlax")
        )
        num = observation.numeric_features[turn_row]
        self.assertAlmostEqual(num[NUMERIC_TT_DAMAGE_FRACTION], 0.30)
        self.assertEqual(num[NUMERIC_TT_CALLED], 1.0)
        self.assertEqual(num[NUMERIC_TM2_PRESENT], 1.0)
        self.assertAlmostEqual(num[NUMERIC_TM2_DAMAGE_FRACTION], 0.15)
        self.assertEqual(vocab.observed_oov_tokens, frozenset())

    def test_negated_second_sub_block_encodes_status_and_species(self) -> None:
        lines = [
            "|player|p1|Alice|",
            "|player|p2|Bob|",
            "|switch|p1a: Golem|Golem, L74|100/100",
            "|switch|p2a: Alakazam|Alakazam, L72|100/100",
            "|turn|1",
            "|move|p1a: Golem|Explosion|p2a: Alakazam",
            "|-damage|p2a: Alakazam|0 fnt",
            "|faint|p1a: Golem",
            "|faint|p2a: Alakazam",
            "|",
            "|switch|p1a: Sandslash|Sandslash, L80|100/100",
            "|switch|p2a: Starmie|Starmie, L76|100/100",
            "|",
            "|upkeep",
            "|turn|2",
        ]
        state = self._state(lines)
        observation = self._encode(state)
        vocab = self._vocab()
        self.assertEqual(len(state.turn_merged_tokens), 3)  # lead + turn + cold pair
        turn_row = TRANSITION_TOKEN_OFFSET + 1
        pair_row = TRANSITION_TOKEN_OFFSET + 2
        self.assertEqual(
            observation.categorical_ids[turn_row][CATEGORY_TM_SECOND_KIND],
            vocab.encode("tt2_status:negated"),
        )
        self.assertEqual(
            observation.categorical_ids[turn_row][CATEGORY_TM_SECOND_SPECIES],
            vocab.encode("tt2_species:Alakazam"),
        )
        self.assertEqual(observation.numeric_features[turn_row][NUMERIC_TM2_PRESENT], 0.0)
        self.assertEqual(
            observation.categorical_ids[pair_row][CATEGORY_SLOT],
            vocab.encode("tt_phase:replacement"),
        )
        self.assertEqual(observation.numeric_features[pair_row][NUMERIC_TM2_PRESENT], 1.0)
        self.assertEqual(vocab.observed_oov_tokens, frozenset())

    def test_budget_counts_merged_tokens(self) -> None:
        # K BUDGET UNIT CHANGE: budget=1 keeps ONE whole-turn token (the most recent),
        # which under per-action semantics would have been ~3 tokens of history.
        state = self._state(self._lines())
        observation = self._encode(state, budget=1)
        self.assertTrue(observation.attention_mask[TRANSITION_TOKEN_OFFSET])
        self.assertFalse(observation.attention_mask[TRANSITION_TOKEN_OFFSET + 1])
        self.assertEqual(
            observation.categorical_ids[TRANSITION_TOKEN_OFFSET][CATEGORY_SLOT],
            self._vocab().encode("tt_phase:turn"),
        )

    def test_v2_2_encode_refuses_a_base_vocabulary(self) -> None:
        # Review MED-2: a vocabulary without the turn-merged families must refuse, not
        # silently OOV-hash all 36+ merged labels.
        from pokezero.randbat_vocab import gen3_category_vocabulary

        base_vocab = gen3_category_vocabulary(SHOWDOWN_ROOT)  # include_turn_merged absent
        state = self._state(self._lines())
        with self.assertRaisesRegex(ValueError, "include_turn_merged"):
            observation_from_player_state(
                state, category_vocab=base_vocab, spec=V2_2_REPLAY_OBSERVATION_SPEC
            )
        # Refusal, not drift: no tt-family token ever reached the OOV band. (The cached
        # base vocab is shared suite-wide, so unrelated fixture species may appear.)
        self.assertFalse(
            {
                token
                for token in base_vocab.observed_oov_tokens
                if token.startswith(("tt_phase:", "tt2_"))
            }
        )

    def test_v2_2_encode_refuses_a_narrowed_categorical_spec(self) -> None:
        # Review MED-3: the categorical census floor (v2.2 is the first categorical
        # width delta; a 39-column v2.2 spec would silently drop the whole second
        # sub-block categorical surface).
        from dataclasses import replace as spec_replace

        state = self._state(self._lines())
        narrowed = spec_replace(
            V2_2_REPLAY_OBSERVATION_SPEC,
            categorical_feature_count=V2_1_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
        )
        with self.assertRaisesRegex(ValueError, "categorical columns"):
            observation_from_player_state(
                state, category_vocab=self._vocab(), spec=narrowed
            )

    def test_v2_2_encode_requires_the_merged_stream(self) -> None:
        replay = parse_showdown_replay(self._lines(), battle_id="turn-merged-encode")
        state = normalize_for_player(
            replay,
            player_id="p1",
            configured_showdown_slot="p1",
            format_id="gen3randombattle",
            # include_turn_merged deliberately omitted.
        )
        with self.assertRaisesRegex(ValueError, "include_turn_merged"):
            observation_from_player_state(
                state,
                category_vocab=self._vocab(),
                spec=V2_2_REPLAY_OBSERVATION_SPEC,
            )


@unittest.skipUnless(
    (SHOWDOWN_ROOT / "data" / "random-battles" / "gen3" / "sets.json").exists(),
    "requires a local Gen 3 Pokemon Showdown checkout",
)
class PinnedBitSurvivesTruncationTest(unittest.TestCase):
    """#514 regression pin: the per-mon pinned Tier-2 CB bit derives from the FULL
    per-action stream and must survive K-truncation of the transition block — verified
    live by the #514 review, pinned here because v2.2 CHANGES the truncation unit
    (tokens are whole turns), so both schemas are asserted side by side."""

    _LINES = [
        "|player|p1|Alice|",
        "|player|p2|Bob|",
        "|switch|p1a: Snorlax|Snorlax, L80|100/100",
        "|switch|p2a: Skarmory|Skarmory, L76|100/100",
        "|turn|1",
        # The (synthetically tier2-concluded) opponent strike, early in the game.
        "|move|p2a: Skarmory|Drill Peck|p1a: Snorlax",
        "|-damage|p1a: Snorlax|85/100",
        "|move|p1a: Snorlax|Body Slam|p2a: Skarmory",
        "|-damage|p2a: Skarmory|70/100",
        "|upkeep",
        "|turn|2",
        # Enough later turns that budget=1 truncates the concluded strike out.
        "|move|p2a: Skarmory|Spikes|p1a: Snorlax",
        "|-sidestart|p1: Alice|Spikes",
        "|move|p1a: Snorlax|Body Slam|p2a: Skarmory",
        "|-damage|p2a: Skarmory|55/100",
        "|upkeep",
        "|turn|3",
        "|move|p2a: Skarmory|Drill Peck|p1a: Snorlax",
        "|-damage|p1a: Snorlax|70/100",
        "|move|p1a: Snorlax|Body Slam|p2a: Skarmory",
        "|-damage|p2a: Skarmory|40/100",
        "|upkeep",
        "|turn|4",
    ]

    @staticmethod
    def _vocab():
        from pokezero.randbat_vocab import gen3_category_vocabulary

        return gen3_category_vocabulary(SHOWDOWN_ROOT, include_turn_merged=True)

    def _annotated_state(self):
        """Annotate the FIRST opponent strike with the as-of-strike CB conclusion, the
        way the env applies Tier2LiveTracker.annotate output, then map it onto the
        merged stream the way the env's v2.2 path does."""
        replay = parse_showdown_replay(self._LINES, battle_id="pinned-truncation")
        state = normalize_for_player(
            replay,
            player_id="p1",
            configured_showdown_slot="p1",
            format_id="gen3randombattle",
            include_turn_merged=True,
        )
        tokens = list(state.transition_tokens)
        first_strike = next(
            index
            for index, token in enumerate(tokens)
            if token.kind == TOKEN_KIND_MOVE and token.actor_slot == "p2"
        )
        tokens[first_strike] = dc_replace(tokens[first_strike], cb_bit=True)
        # ...and the #513 investment code on OUR first strike (turn 1: Snorlax's Body
        # Slam into Skarmory) — as-of-strike, describing the struck defender.
        own_strike = next(
            index
            for index, token in enumerate(tokens)
            if token.kind == TOKEN_KIND_MOVE and token.actor_slot == "p1"
        )
        tokens[own_strike] = dc_replace(tokens[own_strike], investment=1.0)
        annotated = tuple(tokens)
        return dc_replace(
            state,
            transition_tokens=annotated,
            turn_merged_tokens=annotate_turn_merged_tokens(state.turn_merged_tokens, annotated),
        )

    def test_annotations_map_onto_the_merged_sub_blocks(self) -> None:
        state = self._annotated_state()
        # Turn 1: Skarmory (faster) moved first — its FIRST sub-block carries the bit.
        turn_1 = next(token for token in state.turn_merged_tokens if token.turn == 1)
        self.assertEqual(turn_1.first.actor_slot, "p2")
        self.assertTrue(turn_1.first.cb_bit)
        self.assertFalse(turn_1.second.cb_bit)
        # The #513 investment code rode OUR (second-mover) strike of the same turn.
        self.assertEqual(turn_1.second.actor_slot, "p1")
        self.assertEqual(turn_1.second.investment, 1.0)
        self.assertEqual(turn_1.first.investment, 0.0)
        others = [
            sub
            for token in state.turn_merged_tokens
            if token.turn != 1
            for sub in (token.first, token.second)
        ]
        self.assertFalse(any(sub.cb_bit for sub in others))
        self.assertFalse(any(sub.investment for sub in others))

    def test_pinned_bit_survives_truncation_in_both_schemas(self) -> None:
        state = self._annotated_state()
        skarmory_index = next(
            index for index, mon in enumerate(state.opponent_team) if mon.species == "Skarmory"
        )
        for spec in (V2_1_REPLAY_OBSERVATION_SPEC, V2_2_REPLAY_OBSERVATION_SPEC):
            with self.subTest(schema=spec.schema_version):
                observation = observation_from_player_state(
                    state,
                    category_vocab=self._vocab(),
                    spec=spec,
                    feature_masks=ObservationFeatureMasks(
                        transition_token_budget=1, tier2_investment=True
                    ),
                )
                # budget=1 keeps only the LAST transition row; the concluded turn-1
                # strikes are truncated out of the visible history...
                visible_row = observation.numeric_features[TRANSITION_TOKEN_OFFSET]
                self.assertEqual(visible_row[NUMERIC_TT_CB_BIT], 0.0)
                self.assertEqual(visible_row[NUMERIC_TT_INVESTMENT_BIT], 0.0)
                if spec is V2_2_REPLAY_OBSERVATION_SPEC:
                    self.assertEqual(visible_row[NUMERIC_TM2_CB_BIT], 0.0)
                    self.assertEqual(visible_row[NUMERIC_TM2_INVESTMENT], 0.0)
                self.assertFalse(observation.attention_mask[TRANSITION_TOKEN_OFFSET + 1])
                # ...but BOTH per-mon pinned surfaces derive from the FULL per-action
                # stream (CB via actor identity, investment via the #512
                # defender_species channel — which the turn-merge preserves per
                # sub-block) and stand.
                pinned_row = observation.numeric_features[
                    OPPONENT_POKEMON_TOKEN_OFFSET + skarmory_index
                ]
                self.assertEqual(pinned_row[NUMERIC_TIER2_CB_PINNED], 1.0)
                self.assertEqual(pinned_row[NUMERIC_TIER2_INVESTMENT_PINNED], 1.0)

    def test_tier2_mask_darkens_pinned_and_sub_block_bits_under_v2_2(self) -> None:
        state = self._annotated_state()
        observation = observation_from_player_state(
            state,
            category_vocab=self._vocab(),
            spec=V2_2_REPLAY_OBSERVATION_SPEC,
            feature_masks=ObservationFeatureMasks(tier2_residuals=False),
        )
        for row in observation.numeric_features:
            self.assertEqual(row[NUMERIC_TIER2_CB_PINNED], 0.0)
            self.assertEqual(row[NUMERIC_TIER2_INVESTMENT_PINNED], 0.0)
            self.assertEqual(row[NUMERIC_TT_CB_BIT], 0.0)
            self.assertEqual(row[NUMERIC_TM2_CB_BIT], 0.0)
            self.assertEqual(row[NUMERIC_TM2_INVESTMENT], 0.0)


def _second_mover_trio_allowance_keys(merged) -> set[int]:
    """Flattened-stream indices allowed to differ in the trio fields (the documented
    merge): expansions of SECOND sub-blocks under a trio-changing first mover."""
    from pokezero.transitions import (
        SIDE_EFFECT_HAZARD_CLEAR,
        SIDE_EFFECT_HAZARD_SET,
        SIDE_EFFECT_WEATHER_SET,
    )
    from pokezero.turn_merged import SUB_BLOCK_ACTION as _ACTION

    changers = {SIDE_EFFECT_HAZARD_SET, SIDE_EFFECT_HAZARD_CLEAR, SIDE_EFFECT_WEATHER_SET}
    keys: set[int] = set()
    cursor = 0
    for token in merged:
        for position, sub in enumerate((token.first, token.second)):
            if sub.status != _ACTION:
                continue
            expansion = 1 + (1 if sub.cant_reason else 0) + (
                1 if sub.cant_reason and sub.called else 0
            )
            expansion += 1 if sub.baton_pass_species else 0
            if position == 1 and token.first.side_effect in changers:
                keys.update(range(cursor, cursor + expansion))
            cursor += expansion
    return keys


def _live_root():
    import shutil

    from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return root


@unittest.skipUnless(_live_root() is not None, "requires built Showdown checkout and node")
class TurnMergedLivePinnedSweepTest(unittest.TestCase):
    """The pinned↔tracker equality sweep in v2.2 mode (coordinator follow-up to #513/#514).

    Real BattleStream games under a v2.2 env with the tier2 + investment channels live:
    at every final boundary, BOTH per-mon pinned columns (CB 138, investment 139) must
    equal the live trackers' per-mon views EXACTLY — the derivations read the FULL
    per-action stream, whose defender_species channel the turn-merge preserves per
    sub-block — and the annotated merged stream must flatten back to the annotated
    per-action stream field-for-field (residual/validity/cb/investment included)."""

    def test_v2_2_pinned_columns_match_trackers_and_streams_stay_bijective(self) -> None:
        from tests.test_tier2_live_env import _play

        from pokezero.investment import conclusion_column_code
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.turn_merged import flatten_turn_merged_tokens

        root = _live_root()
        pinned_checked = 0
        codes_seen = 0
        bijections_checked = 0
        for seed in (401, 402, 403, 404):
            env = LocalShowdownEnv(
                LocalShowdownConfig(
                    showdown_root=root,
                    set_belief_source=True,
                    observation_spec=V2_2_REPLAY_OBSERVATION_SPEC,
                    feature_masks=ObservationFeatureMasks(tier2_investment=True),
                )
            )
            try:
                _play(env, seed)
                self.assertTrue(env.investment_active())
                for player in ("p1", "p2"):
                    state = env._state_for_player(player)
                    observation = env.observe(player)
                    observation.validate(V2_2_REPLAY_OBSERVATION_SPEC)
                    # Annotated bijection: the merged stream carries EXACTLY the
                    # annotated per-action stream (all four Tier-2-family fields),
                    # modulo the documented second-mover trio merge (same allowance
                    # as the corpus gate: trio fields only, and only under a
                    # trio-changing first mover).
                    flattened = flatten_turn_merged_tokens(state.turn_merged_tokens)
                    self.assertEqual(len(flattened), len(state.transition_tokens))
                    allowance_keys = _second_mover_trio_allowance_keys(
                        state.turn_merged_tokens
                    )
                    for index, (rebuilt, original) in enumerate(
                        zip(flattened, state.transition_tokens)
                    ):
                        if rebuilt == original:
                            continue
                        self.assertIn(
                            index,
                            allowance_keys,
                            f"seed {seed} {player}: non-trio divergence at token {index}",
                        )
                        self.assertEqual(
                            dc_replace(
                                rebuilt,
                                own_spikes_layers=original.own_spikes_layers,
                                opp_spikes_layers=original.opp_spikes_layers,
                                weather=original.weather,
                            ),
                            original,
                            f"seed {seed} {player}: non-trio field diverged at {index}",
                        )
                    bijections_checked += 1
                    codes_seen += sum(
                        1 for token in state.transition_tokens if token.investment
                    )
                    # Pinned ↔ tracker equality, both surfaces.
                    investment_tracker = env._investment_trackers.get(player)
                    expected_investment: dict[str, float] = {}
                    if investment_tracker is not None:
                        for key, conclusion in investment_tracker.conclusions.items():
                            code = conclusion_column_code(conclusion)
                            if code:
                                expected_investment[key.split(":", 1)[1]] = code
                    tier2_tracker = env._tier2_trackers.get(player)
                    expected_cb = set()
                    if tier2_tracker is not None:
                        expected_cb = {
                            key.split(":", 1)[1]
                            for key, concluded in tier2_tracker.cb_bits.items()
                            if concluded
                        }
                    from pokezero.showdown import _normalize_identifier

                    for index, mon in enumerate(state.opponent_team):
                        row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET + index]
                        species = _normalize_identifier(mon.species)
                        self.assertEqual(
                            row[NUMERIC_TIER2_INVESTMENT_PINNED],
                            expected_investment.get(species, 0.0),
                            f"seed {seed} {player} {mon.species}: v2.2 investment pinned drifts",
                        )
                        self.assertEqual(
                            row[NUMERIC_TIER2_CB_PINNED],
                            1.0 if species in expected_cb else 0.0,
                            f"seed {seed} {player} {mon.species}: v2.2 CB pinned drifts",
                        )
                        pinned_checked += 1
            finally:
                env.close()
        # Every seed/perspective must have exercised the full-bench comparison and the
        # annotated bijection. Investment CONCLUSIONS are rare by design (two-strike
        # rule), so codes_seen carries no hard floor — the equality asserts above are
        # the substance either way.
        self.assertGreaterEqual(pinned_checked, 4 * 2 * 6)
        self.assertEqual(bijections_checked, 8)


if __name__ == "__main__":
    unittest.main()
