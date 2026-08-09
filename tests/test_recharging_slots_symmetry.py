"""`_recharging_slots` itself — the production edit, which no test bound.

Review found the risk-bearing change unpinned: every test for this work exercised the HARNESS
helper (`fidelity_gate_events.production_recharging_slots`), and two independent mutations to
the production code left the entire suite green:

  - `if observation_metadata.get("self_must_recharge") is True:` -> `if False:`   0 failures
  - the fallback path returning without prefixing the self lock                   0 failures

The second is exactly the defect `_opponent_recharging_fallback` was extracted to prevent, and
its docstring called that "structurally impossible". Structure is not a test. Both are pinned
here, along with the `"No Move"` mapping the symmetry change made reachable.
"""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero import engine_search  # noqa: E402
from pokezero.engine_search import (  # noqa: E402
    _ENGINE_FORCED_NO_MOVE_IDS,
    EngineMctsPolicy,
    normalize_id,
)


def _context(*, seat: str = "p1", self_mr=None, opp_mr=None, trajectory=None, round_index=None):
    metadata: dict = {}
    if self_mr is not None:
        metadata["self_must_recharge"] = self_mr
    if opp_mr is not None:
        metadata["opponent_must_recharge"] = opp_mr
    return SimpleNamespace(
        player_id=seat,
        observation=SimpleNamespace(metadata=metadata),
        trajectory=trajectory,
        decision_round_index=round_index,
    )


def _policy() -> EngineMctsPolicy:
    policy = EngineMctsPolicy.__new__(EngineMctsPolicy)
    policy.stats = SimpleNamespace(choices_unmapped_causes=Counter(), unmapped_choices=Counter())
    return policy


class SelfSideIsLiveTest(unittest.TestCase):
    """Mutation 1: the symmetric branch."""

    def test_our_own_slot_is_locked_when_the_tracker_says_so(self) -> None:
        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(seat="p1", self_mr=True, opp_mr=False)
        )
        self.assertIn("p1", got, "our own slot must be locked from self_must_recharge")

    def test_it_follows_the_seat_rather_than_being_hardcoded(self) -> None:
        """Non-vacuity: `return ("p1",)` would satisfy the test above."""
        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(seat="p2", self_mr=True, opp_mr=False)
        )
        self.assertEqual(got, ("p2",))

    def test_both_sides_lock_together(self) -> None:
        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(seat="p1", self_mr=True, opp_mr=True)
        )
        self.assertEqual(set(got), {"p1", "p2"})

    def test_a_free_mon_is_not_locked(self) -> None:
        """The other half. Without this, `always lock our slot` would pass."""
        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(seat="p1", self_mr=False, opp_mr=False)
        )
        self.assertNotIn("p1", got)


class TheFallbackCannotDiscardTheSelfLockTest(unittest.TestCase):
    """Mutation 2: the reason the fallback was extracted at all.

    The reconstruction has eleven early `return ()` statements, each meaning "no OPPONENT lock",
    plus one success return. If the self lock is not carried across them, a context that reaches
    the fallback loses it silently. Review mutated exactly this and nothing failed.

    Coverage here is honest rather than complete: the early-return tests all exercise the FIRST
    of the eleven. The success return is covered separately below, because review found a
    mutation that discards the self lock only when the reconstruction succeeds -- surviving all
    24 tests.
    """

    def test_self_lock_survives_when_the_tracker_omits_the_opponent(self) -> None:
        """`self_must_recharge` true, `opponent_must_recharge` ABSENT -> fallback path taken."""
        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(seat="p1", self_mr=True, opp_mr=None)
        )
        self.assertIn("p1", got, "the fallback discarded our own lock")

    def test_self_lock_survives_the_fallbacks_first_early_return(self) -> None:
        """CORRECTED after review: these three shapes all hit the SAME return.

        The docstring used to claim "each shape drives the reconstruction to a different early
        return". Review traced execution and found all three stop at the first of the eleven --
        they differ only in which sub-condition of the same `if` fires. Ten returns are never
        executed by this class.

        One path is sufficient to kill the "drop the prefix at the call site" mutation, because
        the prefix is a single expression -- but it is NOT sufficient in general, which is what
        `test_self_lock_survives_when_the_fallback_SUCCEEDS` below exists to cover.
        """
        shapes = {
            "no trajectory": _context(seat="p1", self_mr=True, round_index=4),
            "no round index": _context(seat="p1", self_mr=True, trajectory=object()),
            "neither": _context(seat="p1", self_mr=True),
        }
        for label, context in shapes.items():
            with self.subTest(shape=label):
                self.assertIn(
                    "p1",
                    EngineMctsPolicy._recharging_slots(_policy(), context),
                    f"self lock lost via the '{label}' return",
                )

    def test_an_explicit_opponent_False_still_keeps_our_lock(self) -> None:
        """An explicit False is proof about the OPPONENT, and must not speak for us."""
        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(seat="p1", self_mr=True, opp_mr=False)
        )
        self.assertEqual(got, ("p1",))

    def test_self_lock_survives_when_the_fallback_SUCCEEDS(self) -> None:
        """The gap review found: a mutation discarding the self lock ONLY on success.

        `_r = fallback(...); return (self_slot + _r) if not _r else _r` survived all 24 tests,
        because every other test in this class drives the reconstruction to an early `return ()`
        -- where `self_slot + ()` and `()`-plus-prefix are indistinguishable. The discard is
        visible only when the reconstruction actually PROVES an opponent lock, which is the
        double-recharge boundary: both actives recharging at once.

        Unreachable on corpus/golden-v4 (both tracker keys present on all 1295 rows, so the
        fallback never runs there), but the fallback exists precisely for pre-pack contexts.
        """
        context = _context(seat="p1", self_mr=True, opp_mr=None, round_index=5)
        context.trajectory = object()
        context.observation.metadata.update(
            {
                "belief_view": {"opponent_pokemon": [{"species": "Snorlax", "active": True}]},
                "recent_public_events": ["|move|p2a: Snorlax|Hyper Beam|p1a: Blissey"],
            }
        )
        action = SimpleNamespace(kind="move", move_id="hyperbeam")
        rounds = {4: SimpleNamespace(actions={"p2": action})}
        with mock.patch.object(
            engine_search, "public_action_rounds_from_trajectory_metadata", lambda _t: rounds
        ):
            got = EngineMctsPolicy._recharging_slots(_policy(), context)
        # Non-vacuity first: if the fallback did NOT prove the opponent lock this test would be
        # exercising an early return again and proving nothing.
        self.assertIn("p2", got, "fallback did not reach its success return -- test is vacuous")
        self.assertIn("p1", got, "the self lock was discarded on the fallback's success path")

    def test_and_a_free_mon_gains_nothing_from_the_fallback(self) -> None:
        """Non-vacuity for the whole class: these paths must still be able to return ()."""
        got = EngineMctsPolicy._recharging_slots(
            _policy(), _context(seat="p1", self_mr=False, opp_mr=None)
        )
        self.assertEqual(got, ())


class ForcedNoMoveMapsTest(unittest.TestCase):
    """The regression symmetry made reachable, and the blocking finding of review.

    The crate displays `MoveChoice::None` as "No Move" for a locked slot; Showdown's request for
    a recharge turn offers exactly one candidate, named `recharge`. Two vocabularies, one forced
    action. Until this mapping existed the decision fell to `_fallback(..., "choices_unmapped")`
    -- a counter engine_search.py:673-675 requires at zero independently of the fallback rate,
    with the cause mislabelled `all_unmapped_legality_mismatch`.

    Before the symmetry change these worlds failed construction earlier (Showdown sets
    `trapped: true` on a recharge request, so `engine_world` rejected them as
    `self_request_state_unsupported`), which is why this was latent rather than new.
    """

    def _recharge_context(self):
        candidates = [{"action_index": 0, "kind": "move", "move_id": "recharge", "legal": True}]
        metadata = {"action_candidates": candidates}
        return SimpleNamespace(
            player_id="p2",
            observation=SimpleNamespace(metadata=metadata, legal_action_mask=[True]),
        )

    def test_no_move_resolves_to_the_requests_recharge_candidate(self) -> None:
        policy = _policy()
        index = EngineMctsPolicy._map_choices(policy, self._recharge_context(), {"No Move": 1.0})
        self.assertEqual(index, 0)
        self.assertEqual(dict(policy.stats.unmapped_choices), {})

    def test_the_bare_engine_token_resolves_too(self) -> None:
        """Defensive only -- and review showed my original justification was wrong.

        I said `depth_tactics_probe` translates the display to "none", so accept that spelling.
        That translation runs display -> `MoveChoice::from_string`, a direction that never
        reaches `_map_choices`. Every producer that DOES reach here (the poke_engine binding and
        lib.rs:87 `move_display`) emits "No Move", so "none" is unreachable today. Kept as a
        cheap guard against a future producer using the bare token, labelled as such rather than
        as a claim about current behaviour.
        """
        policy = _policy()
        self.assertEqual(
            EngineMctsPolicy._map_choices(policy, self._recharge_context(), {"none": 1.0}), 0
        )

    def test_the_display_string_really_does_normalize_into_the_set(self) -> None:
        """Non-vacuity: pins the id the crate emits, so a normalize_id change is visible."""
        self.assertIn(normalize_id("No Move"), _ENGINE_FORCED_NO_MOVE_IDS)

    def test_it_does_not_invent_a_recharge_when_none_is_offered(self) -> None:
        """The mapping must not manufacture a choice on an ordinary turn.

        Without this, `index = move_index_by_id.get("recharge")` returning None would be
        indistinguishable from a mapping that silently picked something.
        """
        candidates = [{"action_index": 0, "kind": "move", "move_id": "return", "legal": True}]
        context = SimpleNamespace(
            player_id="p2",
            observation=SimpleNamespace(
                metadata={"action_candidates": candidates}, legal_action_mask=[True]
            ),
        )
        policy = _policy()
        self.assertIsNone(EngineMctsPolicy._map_choices(policy, context, {"No Move": 1.0}))
        self.assertEqual(dict(policy.stats.unmapped_choices), {"No Move": 1})


def _map_context(candidates, *, mask=None, seat: str = "p1"):
    return SimpleNamespace(
        player_id=seat,
        observation=SimpleNamespace(
            metadata={"action_candidates": candidates},
            legal_action_mask=[True] * len(candidates) if mask is None else mask,
        ),
    )


def _move(index: int, move_id: str, *, legal: bool = True) -> dict:
    return {"action_index": index, "kind": "move", "move_id": move_id, "legal": legal}


def _switch(index: int, species: str) -> dict:
    return {
        "action_index": index,
        "kind": "switch",
        "legal": True,
        "pokemon": {"species": species},
    }


class ForcedNoMoveMapsToStruggleTest(unittest.TestCase):
    """The SECOND vocabulary gap behind the same `MoveChoice::None` token: Struggle.

    Captured, not hypothesised. Battle `bp-trap-lastmon2` d20 seat p1, the first
    `choices_unmapped` record in the corpus whose `world_failures` is EMPTY -- 4 worlds
    constructed AND searched, `engine_choices {"No Move": 4.0}`,
    `request_legal_choices ["struggle"]`, cause mislabelled `all_unmapped_legality_mismatch`,
    then 20 consecutive refusals d20-d39.

    The engine has no Struggle arm: `MoveChoice` is Move/Switch/None and gen3
    `get_all_options` never synthesizes one, so a 0-PP moveset with no live bench falls
    through `add_available_moves` (nothing) and `add_switches` (nothing) to the terminal
    `if options.len() == 0 { push(MoveChoice::None) }`. Showdown at the same state
    substitutes Struggle. `_map_choices` translated `none` to `recharge` only.

    The no-legal-switch precondition is MEASURED on that line, not inferred: d16 and d18
    offered `['struggle', 'switch:Shedinja']` and SEARCHED; d20, same Struggle-only moveset
    with the bench fainted away, REFUSED.
    """

    def test_no_move_resolves_to_the_requests_struggle_candidate(self) -> None:
        policy = _policy()
        context = _map_context([_move(0, "struggle")])
        self.assertEqual(EngineMctsPolicy._map_choices(policy, context, {"No Move": 4.0}), 0)
        self.assertEqual(dict(policy.stats.unmapped_choices), {})
        self.assertEqual(dict(policy.stats.choices_unmapped_causes), {})

    def test_the_struggle_slots_own_action_index_is_returned(self) -> None:
        """The mapped index is the candidate's `action_index`, not its position or a default.

        Originally this docstring claimed to guard the falsy-zero trap (`get(recharge) or
        get(struggle)`). Review measured that claim FALSE: with no recharge candidate the
        `or` chain evaluates `None or 2` -> 2 and this test passes under the mutation. The
        falsy-zero trap only bites when the RECHARGE lookup returns 0, which is
        `test_the_recharge_translation_still_wins_its_own_case` below -- mutant M9 in the
        null-world runner is killed there and nowhere else. Kept, with an honest name: the
        Struggle candidate sits at slot 3 behind two spent slots, so an off-by-one or a
        positional assumption is still visible here.
        """
        policy = _policy()
        context = _map_context(
            [_move(0, "pound", legal=False), _move(1, "tackle", legal=False), _move(2, "struggle")]
        )
        self.assertEqual(EngineMctsPolicy._map_choices(policy, context, {"No Move": 1.0}), 2)

    def test_the_recharge_translation_still_wins_its_own_case(self) -> None:
        """Adding the second lookup must not disturb the first.

        Also the file's ONLY guard against truthiness chaining between the two lookups:
        recharge resolves to action_index 0 here, so `get(recharge) or <struggle>` evaluates
        `0 or None` -> None and the recharge turn stops mapping.
        """
        policy = _policy()
        context = _map_context([_move(0, "recharge")])
        self.assertEqual(EngineMctsPolicy._map_choices(policy, context, {"No Move": 1.0}), 0)
        self.assertEqual(dict(policy.stats.unmapped_choices), {})

    def test_recharge_is_preferred_if_a_request_ever_offered_both(self) -> None:
        """Unreachable by construction -- and now doubly so, which makes the ORDER moot.

        `sim/pokemon.ts` `getMoveRequestData` reaches the Struggle substitution ONLY via
        `else if (!moves.length)`, and a `recharge` lock makes `getMoves` return the
        one-element `[{move: 'Recharge', id: 'recharge'}]`. Non-empty, so the Struggle arm
        cannot run on a recharge turn: the two pseudo-moves are mutually exclusive at the
        single site that emits either. The live `lastmon2` line shows the same from the
        other end -- its requests are `['recharge']` or `['struggle', ...]`, never both.

        The admission gate then enforces that invariant locally: a request carrying both
        has two keys in `move_index_by_id`, so `forced_struggle_index` is None and the
        Struggle arm is inert regardless of which lookup runs first. Swapping the two
        (mutant M3) is consequently EQUIVALENT -- killed by nothing, and proven to change
        no behaviour across a 14-case differential battery, rather than left as an
        unexamined survivor. This test pins the OUTCOME, not the ordering.
        """
        policy = _policy()
        context = _map_context([_move(0, "struggle"), _move(1, "recharge")])
        self.assertEqual(EngineMctsPolicy._map_choices(policy, context, {"No Move": 1.0}), 1)

    def test_it_does_not_invent_a_struggle_when_the_request_offers_real_moves(self) -> None:
        """Neither pseudo-move offered is the ordinary turn, and it must stay unmapped.

        "No Move" against a request with real moves is a genuine engine/request
        disagreement -- the `all_unmapped_legality_mismatch` this class is named for -- and
        papering over it with an invented Struggle would hide a real bug behind a legal action.
        """
        policy = _policy()
        context = _map_context([_move(0, "return"), _switch(1, "Snorlax")])
        self.assertIsNone(EngineMctsPolicy._map_choices(policy, context, {"No Move": 1.0}))
        self.assertEqual(dict(policy.stats.unmapped_choices), {"No Move": 1})

    def test_an_unadmitted_struggle_candidate_is_not_mapped(self) -> None:
        """Both halves of the admission filter, separately.

        Review found the earlier single case named "illegal" while passing `legal=True` with
        a False mask, so the `legal` flag itself was unpinned: mutating the candidate loop's
        `not candidate.get("legal")` to `False` survived every test here AND the end-to-end
        probe. Defence in depth in production -- both fields trace to the same
        `showdown.py:7530` source -- but a test may not assert coverage it does not have.
        """
        for label, legal, mask in (("masked out", True, [False]), ("flagged illegal", False, [True])):
            with self.subTest(label):
                policy = _policy()
                context = _map_context([_move(0, "struggle", legal=legal)], mask=mask)
                self.assertIsNone(
                    EngineMctsPolicy._map_choices(policy, context, {"No Move": 1.0})
                )
                self.assertEqual(dict(policy.stats.unmapped_choices), {"No Move": 1})

    def test_a_struggle_move_slot_beside_a_real_move_is_not_the_pseudo_move(self) -> None:
        """The measured defect of the first revision, and the reason for the count gate.

        Showdown substitutes Struggle only when the request would otherwise offer NO move,
        so a `struggle` candidate beside another legal move is a real Struggle MOVE SLOT.
        Review built exactly that in `gen3customgame` -- a Blissey with
        `moves=("Struggle", "Soft-Boiled")` yields legal `struggle` AND legal `softboiled` --
        and the ungated translation mapped "No Move" onto it, absorbing the genuine
        `all_unmapped_legality_mismatch` this class exists to count.
        """
        policy = _policy()
        context = _map_context(
            [_move(0, "struggle"), _move(1, "softboiled"),
             _move(2, "slot3", legal=False), _move(3, "slot4", legal=False)]
        )
        self.assertIsNone(EngineMctsPolicy._map_choices(policy, context, {"No Move": 1.0}))
        self.assertEqual(dict(policy.stats.unmapped_choices), {"No Move": 1})
        self.assertEqual(
            dict(policy.stats.choices_unmapped_causes), {"all_unmapped_legality_mismatch": 1}
        )

    def test_struggle_beside_a_legal_switch_stays_countable(self) -> None:
        """A legal switch the engine did not propose is a disagreement, and it must stay visible.

        This is the d16/d18 shape off the `lastmon2` line: request `['struggle',
        'switch:Shedinja']`. The engine emitting `MoveChoice::None` there means its world
        saw no switch where the request has one. Translating would keep the decision
        searched and leave NO counter in either world -- `fallback_reasons` never moves
        (the switch arm still maps) and `unmapped_choices` would lose its entry. A campaign
        whose stop condition is `choices_unmapped == 0` cannot afford a path that reaches
        zero by becoming unobservable, so the miss is deliberate.
        """
        policy = _policy()
        context = _map_context([_move(0, "struggle"), _switch(1, "Shedinja")])
        self.assertEqual(
            EngineMctsPolicy._map_choices(
                policy, context, {"No Move": 3.0, "switch shedinja": 1.0}
            ),
            1,
            "the switch the engine DID propose is the only mappable arm here",
        )
        self.assertEqual(dict(policy.stats.unmapped_choices), {"No Move": 1})

    def test_struggle_beside_a_legal_switch_refuses_when_no_move_is_the_only_arm(self) -> None:
        """Same shape, and the price of keeping it countable, stated rather than hidden.

        With "No Move" as the engine's only arm this decision falls back instead of taking
        the one action both sides agree is legal. That is the accepted cost of the
        observability clause above; it is NOT the captured class (`request_legal_choices`
        there is `["struggle"]`, no switch), and it stays counted under its own cause.
        """
        policy = _policy()
        context = _map_context([_move(0, "struggle"), _switch(1, "Shedinja")])
        self.assertIsNone(EngineMctsPolicy._map_choices(policy, context, {"No Move": 1.0}))
        self.assertEqual(
            dict(policy.stats.choices_unmapped_causes), {"all_unmapped_legality_mismatch": 1}
        )

    def test_a_real_move_choice_never_reaches_the_struggle_translation(self) -> None:
        """The translation is keyed on the engine's forced-no-action token, nothing else."""
        policy = _policy()
        context = _map_context([_move(0, "struggle")])
        self.assertIsNone(EngineMctsPolicy._map_choices(policy, context, {"Hyper Beam": 1.0}))
        self.assertEqual(dict(policy.stats.unmapped_choices), {"Hyper Beam": 1})

    def test_a_legal_hidden_power_beside_struggle_blocks_the_translation(self) -> None:
        """Hidden Power is the one move id the map keys twice, so it gets its own case.

        It lands in `move_index_by_id` under its typed id AND in `hidden_power_index`, so
        the one-key admission test already excludes it -- an explicit
        `hidden_power_index is None` clause was written for this and then removed, because
        the null-world runner scored dropping it EQUIVALENT (no behaviour to cover). The
        behaviour is pinned here instead of by an unfalsifiable clause, and this case is
        part of what kills M11 (the count clause weakened to a membership test).
        """
        policy = _policy()
        context = _map_context([_move(0, "struggle"), _move(1, "hiddenpowerice70")])
        self.assertIsNone(EngineMctsPolicy._map_choices(policy, context, {"No Move": 1.0}))
        self.assertEqual(dict(policy.stats.unmapped_choices), {"No Move": 1})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
