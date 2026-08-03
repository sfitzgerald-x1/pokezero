"""Wiring the Tier-2 Choice Band conclusion into the belief candidate sets.

``tests/test_belief_variant_narrowing.py`` covers the belief-side hook in isolation and
``tests/test_investment_belief_narrowing.py`` covers the defender-side producer. This file
covers the ATTACKER-side one — ``Tier2LiveTracker`` handing the engine that mon's Choice
Band variants once the two-strike conclusion stands — and, above all, the SWITCH.

Why the switch matters more than the feature: narrowing moves ``NUMERIC_CANDIDATE_SET_COUNT``
(5) and ``NUMERIC_UNCERTAINTY`` (6), which are frozen legacy positions present in EVERY
schema, plus the possible-items/moves/abilities counts. Turning it on is therefore not an
ablation of something already written — it shifts the input distribution of every checkpoint
ever trained. The byte-identity test below is the guard: with the switch off the encode must
equal the encode of a pipeline whose belief engine never met a tracker, which is exactly what
the pre-wiring encoder was.

The two-strike rule is NOT relaxed here and this file pins that. The investment inference
dropped to a single strike because its test is DEDUCTIVE — an impossible damage roll excludes
a variant outright, so corroboration adds nothing. CB's test is a margin EXCEEDANCE with no
self-validating off-model check: an unmodeled damage booster is indistinguishable from a
Choice Band, so a second independent strike is the only thing standing between one calc edge
case and a permanent, unrecoverable belief exclusion.

Fixtures are the ones tests/test_tier2.py already pins structurally (Snorlax with two
candidate families that differ ONLY in item: Leftovers vs Choice Band).
"""

import unittest

from pokezero.belief import PublicBattleBeliefEngine, belief_key, variant_identity
from pokezero.category_vocab import build_category_vocabulary
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
from pokezero.tier2 import (
    Tier2Config,
    Tier2LiveTracker,
    _choice_band_variant_payloads,
)
from pokezero.transitions import extract_transition_tokens

from tests.test_tier2 import (
    _DEX,
    _OWN_TEAM,
    _SNORLAX_VARIANTS,
    _WHITELIST,
    FakeSource,
    _exceeding_damage,
    _leads,
    _strike_lines,
)

KEY = belief_key("p2", "Snorlax")
_LEFTOVERS_VARIANT, _CB_VARIANT = _SNORLAX_VARIANTS
# Frozen legacy positions, present in every schema — which is exactly why the switch has to
# default off.
_BELIEF_COLUMNS = (
    NUMERIC_CANDIDATE_SET_COUNT,
    NUMERIC_UNCERTAINTY,
    NUMERIC_POSSIBLE_ITEM_COUNT,
)


def _one_strike_lines():
    """A single clean Body Slam exceedance — evidence, but not a conclusion."""

    return _leads() + _strike_lines(_exceeding_damage(), turn=1)


def _concluding_lines():
    """Exactly two independent clean exceedances: the two-strike + non-KO conclusion.

    Ends ON the concluding strike, which is why it is separate from ``_cb_pin_lines``: see
    ``PinTakesEffectOnTheNextRefreshTest``.
    """

    damage = _exceeding_damage()
    # A KO clips the observed value, and the bit requires a NON-KO exceedance.
    assert 330 - 2 * damage > 0, "test setup: neither strike may be a KO"
    lines = _leads()
    lines += _strike_lines(damage, turn=1)
    lines += _strike_lines(damage, turn=2, prior_hp=330 - damage)
    return lines


def _cb_pin_lines():
    """The conclusion PLUS one more strike, so the narrowing has reached the stored belief.

    ``narrow_candidate_variants`` latches a pin; the engine materializes each mon's candidate
    summary when an event updates that mon and re-applies the standing pin there, so a pin
    recorded after the last event of a log has nowhere to land yet. One further turn is the
    realistic live case and is what the belief-surface assertions below need.
    """

    damage = _exceeding_damage()
    return _concluding_lines() + _strike_lines(damage, turn=3, prior_hp=330 - 2 * damage)


def _source():
    return FakeSource({"snorlax": _SNORLAX_VARIANTS})


def _turn_boundaries(lines):
    """The env's observation points: every ``|turn|`` line, plus the full log.

    ``Tier2LiveTracker.annotate`` requires the token stream to align with its own protocol
    fold, so prefixes are taken at action boundaries rather than at arbitrary line counts.
    """

    boundaries = [index + 1 for index, line in enumerate(lines) if line.startswith("|turn|")]
    if not boundaries or boundaries[-1] != len(lines):
        boundaries.append(len(lines))
    return boundaries


def _drive(lines, *, narrow=False, tracker=None, engine=None, fed=0, since=0):
    """Feed ``lines`` to a belief engine + tracker the way a live env does.

    Returns the ingest cursor alongside, so a continuation can pick up where the previous
    call stopped instead of re-ingesting events the engine has already folded. ``since`` is
    the line count a previous call already drove: the tracker's protocol fold is cumulative
    and would reject a prefix shorter than what it has already folded.
    """

    engine = engine or PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=_source())
    if tracker is None:
        tracker = Tier2LiveTracker(
            perspective_slot="p1",
            own_team=_OWN_TEAM,
            dex=_DEX,
            whitelist=_WHITELIST,
            narrow_belief_candidates=narrow,
        )
    for upto in _turn_boundaries(lines):
        if upto <= since:
            continue
        replay = parse_showdown_replay(lines[:upto])
        while fed < len(replay.public_events):
            engine.ingest_event(replay.public_events[fed])
            fed += 1
        tracker.annotate(replay, extract_transition_tokens(replay, perspective_slot="p1"), engine)
    return engine, tracker, fed


def _control_engine(lines):
    """A belief engine fed the same events with NO tracker — the pre-wiring pipeline."""

    engine = PublicBattleBeliefEngine(format_id="gen3randombattle", set_source=_source())
    for event in parse_showdown_replay(lines).public_events:
        engine.ingest_event(event)
    return engine


def _opponent_belief(engine):
    return next(mon for mon in engine.snapshot().side("p2") if mon.species.lower() == "snorlax")


_VOCAB = build_category_vocabulary(
    (
        "species:snorlax",
        "species:slowbro",
        "move:bodyslam",
        "move:earthquake",
        "move:rest",
        "move:sleeptalk",
    )
)


def _encode(lines, engine):
    """Encode under the v2.1 spec.

    The belief columns this file is about (candidate-set count, uncertainty, the possible-*
    counts) are FROZEN LEGACY POSITIONS: identical indices under v2, v2.1 and v2.2, and
    carried into the v3/v4 grouped projections. v2.1 is simply the widest schema whose encode
    needs neither the turn-merged token stream nor the turn-merged vocabulary families, so it
    exercises the same columns without dragging a Showdown checkout into a unit test.

    The tokens are the PLAIN extraction (no tracker annotation), so ``NUMERIC_TIER2_CB_PINNED``
    is 0.0 in both arms of every comparison below — deliberately, since the question here is
    whether the BELIEF surface moved.
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
    def test_the_tracker_defaults_off(self) -> None:
        tracker = Tier2LiveTracker(
            perspective_slot="p1", own_team=_OWN_TEAM, dex=_DEX, whitelist=_WHITELIST
        )
        self.assertFalse(tracker.narrow_belief_candidates)
        self.assertEqual(tracker.belief_narrowing_count, 0)

    def test_cb_rides_the_shared_belief_narrowing_mask(self) -> None:
        """One provenance bit for one class of distribution shift — not a third mask.

        Both tier2 producers mutate the same Tier-1 candidate sets and perturb the same
        frozen legacy belief columns, so a cache whose metadata says narrowing was off has
        to mean NEITHER of them ran.
        """

        self.assertFalse(ObservationFeatureMasks().investment_belief_narrowing)


class TwoStrikeRuleIsUnchangedTest(unittest.TestCase):
    """CB's exceedance test is not the investment lattice; corroboration is load-bearing."""

    def test_the_config_still_requires_two_strikes(self) -> None:
        self.assertEqual(Tier2Config().required_cb_strikes, 2)

    def test_one_clean_exceedance_narrows_nothing(self) -> None:
        engine, tracker, _ = _drive(_one_strike_lines(), narrow=True)
        self.assertFalse(tracker.cb_bits.get(KEY, False))
        self.assertEqual(tracker.belief_narrowing_count, 0)
        self.assertEqual(engine._variant_pins, {})
        # ...and the belief is exactly the untracked one: a lone exceedance leaves no trace.
        self.assertEqual(
            len(_opponent_belief(engine).candidate_variants),
            len(_opponent_belief(_control_engine(_one_strike_lines())).candidate_variants),
        )


class SwitchOffIsByteIdenticalTest(unittest.TestCase):
    """The load-bearing guard: switch off == a belief engine that never met a tracker."""

    def test_switch_off_records_no_narrowing_and_leaves_the_belief_untouched(self) -> None:
        lines = _cb_pin_lines()
        engine, tracker, _ = _drive(lines, narrow=False)
        # The conclusion IS reached — this is not a vacuous pass.
        self.assertTrue(tracker.cb_bits.get(KEY))
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
        lines = _cb_pin_lines()
        engine, tracker, _ = _drive(lines, narrow=False)
        self.assertTrue(tracker.cb_bits.get(KEY))  # non-vacuous
        tracked = _encode(lines, engine)
        untracked = _encode(lines, _control_engine(lines))
        self.assertEqual(tracked.numeric_features, untracked.numeric_features)
        self.assertEqual(tracked.categorical_ids, untracked.categorical_ids)


class SwitchOnNarrowsTest(unittest.TestCase):
    def test_the_conclusion_narrows_to_the_choice_band_family(self) -> None:
        lines = _cb_pin_lines()
        engine, tracker, _ = _drive(lines, narrow=True)
        self.assertTrue(tracker.cb_bits.get(KEY))
        self.assertGreaterEqual(tracker.belief_narrowing_count, 1)
        self.assertEqual(engine._variant_pins[KEY], frozenset({variant_identity(_CB_VARIANT)}))

        belief = _opponent_belief(engine)
        self.assertEqual(len(belief.candidate_variants), 1)
        self.assertEqual(belief.candidate_variants[0]["variant_id"], "lax-2")
        control = _opponent_belief(_control_engine(lines))
        self.assertLess(belief.candidate_set_count, control.candidate_set_count)
        self.assertLess(belief.uncertainty, control.uncertainty)
        # The item surface is now a certainty, which is the whole point: a first-class belief
        # field, not a reserved bit sitting beside an unchanged candidate set.
        self.assertEqual(tuple(belief.possible_items), ("Choice Band",))
        self.assertIn("Leftovers", control.possible_items)

    def test_narrowing_moves_the_encoded_belief_columns(self) -> None:
        lines = _cb_pin_lines()
        narrowed, _, _ = _drive(lines, narrow=True)
        plain, _, _ = _drive(lines, narrow=False)
        row = _encode(lines, narrowed).numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        baseline = _encode(lines, plain).numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET]
        self.assertNotEqual(row, baseline)
        for column in _BELIEF_COLUMNS:
            self.assertLess(row[column], baseline[column], f"column {column}")
        # Both families carry the same four moves, so the move count correctly does NOT move:
        # the narrowing drops only what no surviving variant can have.
        self.assertEqual(
            row[NUMERIC_POSSIBLE_MOVE_COUNT], baseline[NUMERIC_POSSIBLE_MOVE_COUNT]
        )

    def test_later_strikes_re_push_the_same_pin_without_double_applying(self) -> None:
        """The third strike re-derives survivors from the narrowed set and changes nothing.

        `_narrow` runs on every assessed strike whose conclusion stands, not once at the
        concluding one — that is what makes it self-healing. The count proves the repeat
        pushes are no-ops rather than a pin that keeps being rewritten.
        """

        engine, tracker, _ = _drive(_cb_pin_lines(), narrow=True)
        self.assertEqual(tracker.belief_narrowing_count, 1)
        self.assertEqual(engine.variant_pin_conflicts, {})
        self.assertEqual(engine._variant_pins[KEY], frozenset({variant_identity(_CB_VARIANT)}))


class PinTakesEffectOnTheNextRefreshTest(unittest.TestCase):
    """``narrow_candidate_variants`` LATCHES a pin; it does not rewrite stored beliefs.

    The engine materializes a mon's candidate summary when an event updates that mon, and
    re-applies the standing pin at that point. So a pin recorded after the last event of a
    prefix is real and durable but not yet visible on the stored belief — it lands on the
    next reveal that touches the mon. Pinned here because it is shared, load-bearing
    behaviour of the hook (the investment producer has it too), not something this wiring
    introduced, and because it is the reason the fixtures above carry a trailing turn.
    """

    def test_the_pin_is_recorded_immediately_and_applied_at_the_next_reveal(self) -> None:
        concluding = _concluding_lines()
        engine, tracker, fed = _drive(concluding, narrow=True)
        self.assertTrue(tracker.cb_bits.get(KEY))
        self.assertEqual(tracker.belief_narrowing_count, 1)
        self.assertEqual(engine._variant_pins[KEY], frozenset({variant_identity(_CB_VARIANT)}))
        # Recorded, but the stored belief still predates it.
        self.assertEqual(len(_opponent_belief(engine).candidate_variants), 2)

        _drive(_cb_pin_lines(), narrow=True, tracker=tracker, engine=engine, fed=fed,
               since=len(concluding))
        self.assertEqual(len(_opponent_belief(engine).candidate_variants), 1)
        # Still one narrowing: the later strike re-pushed the identical pin.
        self.assertEqual(tracker.belief_narrowing_count, 1)
        self.assertEqual(engine.variant_pin_conflicts, {})


class SurvivorComputationTest(unittest.TestCase):
    def test_only_choice_band_variants_survive(self) -> None:
        survivors = _choice_band_variant_payloads(_SNORLAX_VARIANTS)
        self.assertEqual([v["variant_id"] for v in survivors], ["lax-2"])

    def test_survivors_are_the_engines_own_payload_objects(self) -> None:
        """Identity matching only works if both sides see the SAME mappings."""

        survivors = _choice_band_variant_payloads(_SNORLAX_VARIANTS)
        self.assertIs(survivors[0], _CB_VARIANT)

    def test_no_choice_band_candidate_yields_an_empty_list_not_a_wrong_one(self) -> None:
        """Reachable behind a standing conclusion: Tier-1 keeps shrinking after it freezes.

        The producer must hand back nothing rather than invent a survivor; the engine's
        refusal asymmetry then declines to narrow instead of eliminating everything.
        """

        self.assertEqual(_choice_band_variant_payloads([_LEFTOVERS_VARIANT]), ())
        self.assertEqual(_choice_band_variant_payloads([]), ())

    def test_the_item_predicate_is_the_assessments_own_normalization(self) -> None:
        """Same ``normalize_id`` split ``_assess_strike`` takes ``max_cb_hp`` over."""

        spellings = [
            dict(_CB_VARIANT, variant_id="spaced", item="Choice Band"),
            dict(_CB_VARIANT, variant_id="bare", item="choiceband"),
            dict(_CB_VARIANT, variant_id="shouty", item="CHOICE BAND"),
            dict(_CB_VARIANT, variant_id="none", item=None),
            dict(_CB_VARIANT, variant_id="scarf", item="Choice Scarf"),
        ]
        self.assertEqual(
            [v["variant_id"] for v in _choice_band_variant_payloads(spellings)],
            ["spaced", "bare", "shouty"],
        )


class SearchCloneTest(unittest.TestCase):
    """Both the tracker and the engine are cloned per search branch."""

    def test_a_branch_inherits_the_narrowing_without_reapplying_it(self) -> None:
        lines = _cb_pin_lines()
        engine, tracker, fed = _drive(lines, narrow=True)
        narrowings = tracker.belief_narrowing_count

        branch_engine = engine.clone()
        branch_tracker = tracker.clone()
        self.assertTrue(branch_tracker.narrow_belief_candidates)
        self.assertEqual(branch_tracker.belief_narrowing_count, narrowings)
        self.assertEqual(branch_engine._variant_pins, engine._variant_pins)

        # Re-annotating the same replay in the branch must be a no-op: every strike is already
        # assessed, the tracker's ledger is frozen, and re-deriving the Choice Band variants
        # from the already-narrowed set reproduces the same pin — so nothing double-applies and
        # no conflict is counted.
        replay = parse_showdown_replay(lines)
        branch_tracker.annotate(
            replay, extract_transition_tokens(replay, perspective_slot="p1"), branch_engine
        )
        self.assertEqual(branch_tracker.belief_narrowing_count, narrowings)
        self.assertEqual(branch_engine.variant_pin_conflicts, {})
        self.assertEqual(len(_opponent_belief(branch_engine).candidate_variants), 1)

    def test_a_branch_that_continues_from_before_the_pin_re_narrows_its_own_engine(self) -> None:
        """A snapshot taken pre-conclusion must be able to reach the pin, not lose it."""

        prefix = _one_strike_lines()
        engine, tracker, fed = _drive(prefix, narrow=True)
        self.assertEqual(engine._variant_pins, {})

        branch_engine = engine.clone()
        branch_tracker = tracker.clone()
        _drive(
            _cb_pin_lines(),
            narrow=True,
            tracker=branch_tracker,
            engine=branch_engine,
            fed=fed,
            since=len(prefix),
        )
        self.assertEqual(branch_tracker.belief_narrowing_count, 1)
        self.assertEqual(len(_opponent_belief(branch_engine).candidate_variants), 1)
        # The parent's engine is untouched — a sibling branch still sees both families.
        self.assertEqual(engine._variant_pins, {})
        self.assertEqual(len(_opponent_belief(engine).candidate_variants), 2)

    def test_a_branch_keeps_the_pin_when_its_mon_never_strikes_again(self) -> None:
        """The pin travels on the ENGINE clone; re-narrowing is a repair, not the mechanism.

        Self-healing only reaches mons that get assessed again, so the engine carrying its
        own pins is what makes a leaf agree with its root.
        """

        lines = _cb_pin_lines()
        engine, tracker, _ = _drive(lines, narrow=True)
        branch_engine = engine.clone()
        # No further annotate call at all — the branch simply reads its belief.
        self.assertEqual(branch_engine._variant_pins[KEY], engine._variant_pins[KEY])
        self.assertEqual(len(_opponent_belief(branch_engine).candidate_variants), 1)
        self.assertEqual(tuple(_opponent_belief(branch_engine).possible_items), ("Choice Band",))

    def test_a_rebuilt_tracker_keeps_the_belief_but_not_the_column_conclusion(self) -> None:
        """The narrowing is one-way for the COLUMN, and that is by design, not a bug.

        ``LocalShowdownEnv`` has restore paths that install a belief engine and then build
        fresh trackers (a snapshot with no annotation cache, the synthetic-belief path). A
        fresh tracker re-assessing against an ALREADY-NARROWED engine sees no non-Choice-Band
        variant left, so every strike reports ``cb-pinned-by-elimination`` and the two-strike
        ledger never refills: ``cb_bits`` comes back empty and ``NUMERIC_TIER2_CB_PINNED``
        would encode 0.0 under v2.1/v2.2/v3.

        That is the deliberate Tier-1 boundary (``test_cb_pinned_by_elimination_is_left_to
        _tier1``): once the candidate set alone pins the item, the tier2 exceedance bit stands
        down. The BELIEF is the surface that carries the conclusion forward, it rides the
        engine clone, and it is the only surface v4 has — so nothing is lost where this
        switch is meant to be used. Pinned here so the interaction is visible rather than
        discovered later as column flicker.
        """

        lines = _cb_pin_lines()
        engine, tracker, _ = _drive(lines, narrow=True)
        self.assertTrue(tracker.cb_bits.get(KEY))

        rebuilt = Tier2LiveTracker(
            perspective_slot="p1",
            own_team=_OWN_TEAM,
            dex=_DEX,
            whitelist=_WHITELIST,
            narrow_belief_candidates=True,
        )
        replay = parse_showdown_replay(lines)
        rebuilt.annotate(
            replay, extract_transition_tokens(replay, perspective_slot="p1"), engine
        )
        self.assertEqual(rebuilt.cb_bits, {})
        self.assertEqual(rebuilt.belief_narrowing_count, 0)
        # ...and the belief conclusion is intact, with no conflict counted.
        self.assertEqual(len(_opponent_belief(engine).candidate_variants), 1)
        self.assertEqual(tuple(_opponent_belief(engine).possible_items), ("Choice Band",))
        self.assertEqual(engine.variant_pin_conflicts, {})


if __name__ == "__main__":
    unittest.main()
