"""Wiring the defender-side investment conclusions into the belief candidate sets.

``tests/test_belief_variant_narrowing.py`` covers the belief-side hook in isolation. This
file covers the PRODUCER — ``InvestmentLiveTracker`` computing survivors through the shared
gen3 spread core and pushing them into the caller's engine — and, above all, the SWITCH.

Why the switch matters more than the feature: narrowing moves ``NUMERIC_CANDIDATE_SET_COUNT``
(5) and ``NUMERIC_UNCERTAINTY`` (6), which are frozen legacy positions present in EVERY
schema, plus the possible-items/moves/abilities counts. Turning it on is therefore not an
ablation of something already written — it shifts the input distribution of every checkpoint
ever trained. The byte-identity test below is the guard: with the switch off the encode must
equal the encode of a pipeline that has no investment tracker at all, which is exactly what
the pre-wiring encoder was.

Fixtures are the ones tests/test_investment.py already pins structurally (Slowbro TRIMMED
281 / FULL 282 max HP, def 221 vs 222).
"""

import unittest

from pokezero.belief import PublicBattleBeliefEngine, belief_key, variant_identity
from pokezero.category_vocab import build_category_vocabulary
from pokezero.investment import (
    InvestmentConclusion,
    InvestmentLiveTracker,
    _surviving_variant_payloads,
)
from pokezero.observation import ObservationFeatureMasks
from pokezero.showdown import (
    NUMERIC_CANDIDATE_SET_COUNT,
    NUMERIC_POSSIBLE_ITEM_COUNT,
    NUMERIC_POSSIBLE_MOVE_COUNT,
    NUMERIC_UNCERTAINTY,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    V2_1_REPLAY_OBSERVATION_SPEC,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
)
from pokezero.transitions import extract_transition_tokens

from tests.test_investment import (
    _DEX,
    _FULL_HP,
    _OWN_TEAM,
    _TRIMMED_HP,
    _VARIANT_DEF_FULL,
    _VARIANT_DEF_REDUCED,
    _VARIANT_FULL,
    _VARIANT_TRIMMED,
    FakeSource,
    _leads,
    _strike_lines,
)

KEY = belief_key("p2", "Slowbro")
# Frozen legacy positions, present in every schema — which is exactly why the switch has to
# default off.
_BELIEF_COLUMNS = (
    NUMERIC_CANDIDATE_SET_COUNT,
    NUMERIC_UNCERTAINTY,
    NUMERIC_POSSIBLE_MOVE_COUNT,
)


def _trimmed_pin_lines():
    """Two clean Double-Edge strikes that pin the TRIMMED (Belly Drum) max HP."""

    lines = _leads(_TRIMMED_HP)
    lines += _strike_lines(130, turn=1, max_hp=_TRIMMED_HP)
    lines += _strike_lines(130, turn=2, max_hp=_TRIMMED_HP, prior_hp=_TRIMMED_HP - 130)
    return lines


def _source():
    return FakeSource({"slowbro": [_VARIANT_TRIMMED, _VARIANT_FULL]})


def _drive(lines, *, narrow, tracker=None, engine=None):
    """Feed ``lines`` to a belief engine + tracker the way a live env does."""

    engine = engine or PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=_source())
    if tracker is None:
        tracker = InvestmentLiveTracker(
            perspective_slot="p1",
            own_team=_OWN_TEAM,
            dex=_DEX,
            narrow_belief_candidates=narrow,
        )
    fed = 0
    for upto in (5, 10, len(lines)):
        replay = parse_showdown_replay(lines[:upto])
        while fed < len(replay.public_events):
            engine.ingest_event(replay.public_events[fed])
            fed += 1
        tracker.observe(replay, extract_transition_tokens(replay, perspective_slot="p1"), engine)
    return engine, tracker


def _control_engine(lines):
    """A belief engine fed the same events with NO tracker — the pre-wiring pipeline."""

    engine = PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=_source())
    for event in parse_showdown_replay(lines).public_events:
        engine.ingest_event(event)
    return engine


def _opponent_belief(engine):
    return next(mon for mon in engine.snapshot().side("p2") if mon.species.lower() == "slowbro")


_VOCAB = build_category_vocabulary(
    ("species:slowbro", "species:flareon", "move:doubleedge", "move:surf", "move:bellydrum")
)


def _encode(lines, engine):
    """Encode under the v2.1 spec.

    The belief columns this file is about (candidate-set count, uncertainty, the possible-*
    counts) are FROZEN LEGACY POSITIONS: identical indices under v2, v2.1 and v2.2, and
    carried into the v3/v4 grouped projections. v2.1 is simply the widest schema whose encode
    needs neither the turn-merged token stream nor the turn-merged vocabulary families, so it
    exercises the same columns without dragging a Showdown checkout into a unit test.
    """

    state = normalize_for_player(
        parse_showdown_replay(lines),
        player_id="p1",
        configured_showdown_slot="p1",
        format_id="gen3randombattle",
        belief_engine=engine,
    )
    return observation_from_player_state(
        state, category_vocab=_VOCAB, dex=_DEX, spec=V2_1_REPLAY_OBSERVATION_SPEC
    )


class SwitchDefaultTest(unittest.TestCase):
    def test_the_mask_defaults_off(self) -> None:
        self.assertFalse(ObservationFeatureMasks().investment_belief_narrowing)

    def test_the_tracker_defaults_off(self) -> None:
        tracker = InvestmentLiveTracker(perspective_slot="p1", own_team=_OWN_TEAM, dex=_DEX)
        self.assertFalse(tracker.narrow_belief_candidates)


class SwitchOffIsByteIdenticalTest(unittest.TestCase):
    """The load-bearing guard: switch off == no investment tracker at all."""

    def test_switch_off_records_no_narrowing_and_leaves_the_belief_untouched(self) -> None:
        lines = _trimmed_pin_lines()
        engine, tracker = _drive(lines, narrow=False)
        # The conclusion IS reached — this is not a vacuous pass.
        self.assertEqual(tracker.conclusions[KEY].hp_value, _TRIMMED_HP)
        self.assertEqual(tracker.belief_narrowing_count, 0)
        self.assertEqual(engine._variant_pins, {})
        self.assertEqual(engine.variant_pin_conflicts, {})

        belief = _opponent_belief(engine)
        control = _opponent_belief(_control_engine(lines))
        self.assertEqual(belief.candidate_set_count, control.candidate_set_count)
        self.assertEqual(belief.uncertainty, control.uncertainty)
        self.assertEqual(belief.candidate_variants, control.candidate_variants)
        self.assertEqual(belief.possible_items, control.possible_items)
        self.assertEqual(belief.possible_moves, control.possible_moves)

    def test_switch_off_encodes_byte_identically_to_the_untracked_pipeline(self) -> None:
        lines = _trimmed_pin_lines()
        engine, _ = _drive(lines, narrow=False)
        tracked = _encode(lines, engine)
        untracked = _encode(lines, _control_engine(lines))
        self.assertEqual(tracked.numeric_features, untracked.numeric_features)
        self.assertEqual(tracked.categorical_ids, untracked.categorical_ids)


class SwitchOnNarrowsTest(unittest.TestCase):
    def test_an_hp_conclusion_narrows_to_the_surviving_family(self) -> None:
        lines = _trimmed_pin_lines()
        engine, tracker = _drive(lines, narrow=True)
        self.assertEqual(tracker.conclusions[KEY].hp_value, _TRIMMED_HP)
        self.assertGreaterEqual(tracker.belief_narrowing_count, 1)
        self.assertEqual(
            engine._variant_pins[KEY], frozenset({variant_identity(_VARIANT_TRIMMED)})
        )

        belief = _opponent_belief(engine)
        self.assertEqual(len(belief.candidate_variants), 1)
        self.assertEqual(belief.candidate_variants[0]["variant_id"], "bro-trim")
        control = _opponent_belief(_control_engine(lines))
        self.assertLess(belief.candidate_set_count, control.candidate_set_count)
        self.assertLess(belief.uncertainty, control.uncertainty)
        # The narrowing reaches the surfaces the reserved column cannot: Belly Drum is now
        # a certainty and Surf is excluded.
        self.assertIn("bellydrum", belief.possible_moves)
        self.assertNotIn("surf", belief.possible_moves)

    def test_narrowing_moves_the_encoded_belief_columns(self) -> None:
        lines = _trimmed_pin_lines()
        narrowed, _ = _drive(lines, narrow=True)
        plain, _ = _drive(lines, narrow=False)
        row = _encode(lines, narrowed).numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        baseline = _encode(lines, plain).numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertNotEqual(row, baseline)
        for column in _BELIEF_COLUMNS:
            self.assertLess(row[column], baseline[column], f"column {column}")
        # Both families carry Leftovers, so the item count correctly does NOT move: the
        # narrowing drops only what no surviving variant can have.
        self.assertEqual(
            row[NUMERIC_POSSIBLE_ITEM_COUNT], baseline[NUMERIC_POSSIBLE_ITEM_COUNT]
        )

    def test_a_defense_conclusion_narrows_within_the_hp_family(self) -> None:
        """Def pins separate variants that share a max HP, so HP alone cannot do it."""

        source = FakeSource({"slowbro": [_VARIANT_DEF_REDUCED, _VARIANT_DEF_FULL]})
        lines = _leads(_FULL_HP)
        lines += _strike_lines(131, turn=1, max_hp=_FULL_HP)  # unique to the IV-30 lattice
        engine = PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=source)
        engine, tracker = _drive(lines, narrow=True, engine=engine)
        self.assertEqual(tracker.conclusions[KEY].defense_values, {"def": 221})
        self.assertIsNone(tracker.conclusions[KEY].hp_value)
        self.assertEqual(
            engine._variant_pins[KEY], frozenset({variant_identity(_VARIANT_DEF_REDUCED)})
        )


class SurvivorComputationTest(unittest.TestCase):
    def _survivors(self, conclusion, variants):
        return _surviving_variant_payloads(
            conclusion=conclusion,
            candidate_variants=variants,
            species_key="slowbro",
            dex=_DEX,
            spread_cache={},
        )

    def test_no_conclusion_yields_no_survivors(self) -> None:
        self.assertEqual(
            self._survivors(InvestmentConclusion(KEY), [_VARIANT_TRIMMED, _VARIANT_FULL]), ()
        )

    def test_axes_intersect(self) -> None:
        # HP 282 alone keeps both defense variants; adding def=221 keeps only the IV-30 one.
        both = self._survivors(
            InvestmentConclusion(KEY, hp_value=_FULL_HP),
            [_VARIANT_DEF_REDUCED, _VARIANT_DEF_FULL],
        )
        self.assertEqual(len(both), 2)
        intersected = self._survivors(
            InvestmentConclusion(KEY, hp_value=_FULL_HP, defense_values={"def": 221}),
            [_VARIANT_DEF_REDUCED, _VARIANT_DEF_FULL],
        )
        self.assertEqual([v["variant_id"] for v in intersected], ["bro-hpbug"])

    def test_survivors_are_the_engines_own_payload_objects(self) -> None:
        """Identity matching only works if both sides see the SAME mappings."""

        variants = [_VARIANT_TRIMMED, _VARIANT_FULL]
        survivors = self._survivors(InvestmentConclusion(KEY, hp_value=_TRIMMED_HP), variants)
        self.assertEqual(len(survivors), 1)
        self.assertIs(survivors[0], _VARIANT_TRIMMED)

    def test_a_contradictory_conclusion_yields_an_empty_list_not_a_wrong_one(self) -> None:
        # No variant has this max HP. The producer must hand back nothing rather than
        # guess; the engine's refusal asymmetry then declines to narrow.
        self.assertEqual(
            self._survivors(
                InvestmentConclusion(KEY, hp_value=_TRIMMED_HP + 500),
                [_VARIANT_TRIMMED, _VARIANT_FULL],
            ),
            (),
        )

    def test_an_unevaluable_variant_disables_the_whole_narrowing(self) -> None:
        """An unevaluable candidate could be the true one, so no exclusion is safe."""

        unknown = dict(_VARIANT_FULL, variant_id="bro-unknown")
        self.assertEqual(
            _surviving_variant_payloads(
                conclusion=InvestmentConclusion(KEY, hp_value=_TRIMMED_HP),
                candidate_variants=[_VARIANT_TRIMMED, unknown],
                species_key="notaspecies",  # dex lookup fails -> spread is None
                dex=_DEX,
                spread_cache={},
            ),
            (),
        )


class SearchCloneTest(unittest.TestCase):
    """Both the tracker and the engine are cloned per search branch."""

    def test_a_branch_inherits_the_narrowing_without_reapplying_it(self) -> None:
        lines = _trimmed_pin_lines()
        engine, tracker = _drive(lines, narrow=True)
        narrowings = tracker.belief_narrowing_count

        branch_engine = engine.clone()
        branch_tracker = tracker.clone()
        self.assertTrue(branch_tracker.narrow_belief_candidates)
        self.assertEqual(branch_tracker.belief_narrowing_count, narrowings)
        self.assertEqual(branch_engine._variant_pins, engine._variant_pins)

        # Re-observing the same replay in the branch must be a no-op: the tracker's
        # ledger is frozen and re-deriving survivors from the already-narrowed set
        # reproduces the same pin, so nothing double-applies and no conflict is counted.
        replay = parse_showdown_replay(lines)
        branch_tracker.observe(
            replay, extract_transition_tokens(replay, perspective_slot="p1"), branch_engine
        )
        self.assertEqual(branch_tracker.belief_narrowing_count, narrowings)
        self.assertEqual(branch_engine.variant_pin_conflicts, {})
        self.assertEqual(len(_opponent_belief(branch_engine).candidate_variants), 1)

    def test_a_branch_that_continues_from_before_the_pin_re_narrows_its_own_engine(self) -> None:
        """A snapshot taken pre-conclusion must be able to re-derive the pin, not lose it."""

        prefix = _leads(_TRIMMED_HP)
        engine, tracker = _drive(prefix, narrow=True)
        self.assertEqual(engine._variant_pins, {})

        branch_engine = engine.clone()
        branch_tracker = tracker.clone()
        _drive(
            _trimmed_pin_lines(), narrow=True, tracker=branch_tracker, engine=branch_engine
        )
        self.assertEqual(branch_tracker.belief_narrowing_count, 1)
        self.assertEqual(len(_opponent_belief(branch_engine).candidate_variants), 1)
        # The parent's engine is untouched — a sibling branch still sees both families.
        self.assertEqual(engine._variant_pins, {})
        self.assertEqual(len(_opponent_belief(engine).candidate_variants), 2)


if __name__ == "__main__":
    unittest.main()
