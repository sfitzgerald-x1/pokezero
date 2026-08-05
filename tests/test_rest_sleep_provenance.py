"""Rest-sleep provenance, BUILD half: attempt count -> payload row -> ``rest_turns``.

The export half (``ShowdownReplayState.rest_sleep_counts``, covered in
``tests/test_observation_spec_v3.py``) records, per sleeping mon, that the sleep came
from its own Rest and how many move ATTEMPTS have already burned off it. This file
covers everything downstream of that: direct-world row annotation, ability-aware
counter reconstruction, the capability gate, and the claim that every input is
public-line-derived.

Ground truth for the behaviour all of this exists to reproduce is
``scripts/gen3_switch_differential.py --only hypnosisrestclause hypnosisrestclausecontrol``
(real Node sim), and the engine contract is pinned natively by
``rust/pokezero-search/tests/gen3_rest_sleep_clause.rs``.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero import engine_world  # noqa: E402
from pokezero.dex import MoveInfo, ShowdownDex, SpeciesInfo  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    battle_spec_from_payload,
)
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.gen3_damage import gen3_hp_stat  # noqa: E402
from pokezero.local_showdown import _apply_rest_sleep_provenance  # noqa: E402
from pokezero.showdown import _ReplayParser, parse_showdown_replay  # noqa: E402
from pokezero.showdown_fixture import FixturePokemon, pack_team  # noqa: E402


@contextlib.contextmanager
def _stubbed_capability_probe():
    """Swap the module-level Rest capability probe for the duration.

    The real probe asks the installed native wheel whether it round-trips
    ``rest_turns``. These tests are about the WIRING either side of it -- the
    payload row in, the ``rest_turns`` out -- so they stub it, exactly as
    ``tests/test_engine_world.py`` does for move-trapping. Without the stub a
    machine with no wheel would ERROR here instead of skipping, and the failure
    would name the wrong thing. ``RestTurnsCapabilityTests`` below runs the real
    probe against the real wheel and skips loudly when there is none.
    """

    import pokezero.engine_world as engine_world

    calls: list[int] = []
    original = engine_world.require_rest_turns_support
    engine_world.require_rest_turns_support = lambda *a, **k: calls.append(1)
    try:
        yield calls
    finally:
        engine_world.require_rest_turns_support = original


# --- fixtures ---------------------------------------------------------------------

def _dex() -> ShowdownDex:
    def species(species_id, name, types, base, weight):
        return SpeciesInfo(id=species_id, name=name, types=types, base_stats=base, weight_kg=weight)

    def move(move_id, pp):
        return MoveInfo(
            id=move_id, name=move_id, type="normal", category="physical",
            gen3_category="physical", base_power=50, accuracy=100.0, priority=0,
            recoil=False, drain=False, heal=False, status=None, boosts={},
            target="normal", selfdestruct=False, pp=pp,
        )

    return ShowdownDex(
        moves={"bodyslam": move("bodyslam", 15), "rest": move("rest", 10), "surf": move("surf", 15)},
        species={
            "snorlax": species("snorlax", "Snorlax", ("normal",),
                               {"hp": 160, "atk": 110, "def": 65, "spa": 65, "spd": 110, "spe": 30}, 460.0),
            "skarmory": species("skarmory", "Skarmory", ("steel", "flying"),
                                {"hp": 65, "atk": 80, "def": 140, "spa": 40, "spd": 70, "spe": 70}, 50.5),
            "starmie": species("starmie", "Starmie", ("water", "psychic"),
                               {"hp": 60, "atk": 75, "def": 85, "spa": 100, "spd": 85, "spe": 115}, 80.0),
            "unown": species("unown", "Unown-Z", ("psychic",),
                               {"hp": 48, "atk": 72, "def": 48, "spa": 72, "spd": 48, "spe": 48}, 5.0),
        },
        type_chart={},
    )


_SNORLAX = FixturePokemon(species="Snorlax", moves=("bodyslam",), ability="Immunity",
                          item="Leftovers", level=80,
                          evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
_SKARMORY = FixturePokemon(species="Skarmory", moves=("rest",), ability="Keen Eye",
                           item="Leftovers", level=76,
                           evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
_SKARMORY_EARLY_BIRD = FixturePokemon(species="Skarmory", moves=("rest",), ability="Early Bird",
                                      item="Leftovers", level=76,
                                      evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
_UNOWN_Z = FixturePokemon(species="Unown-Z", moves=("rest",), ability="Levitate",
                          item="Leftovers", level=76,
                          evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
_UNOWN_QUESTION = FixturePokemon(species="Unown-Question", moves=("rest",), ability="Levitate",
                                 item="Leftovers", level=76,
                                 evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})
_STARMIE = FixturePokemon(species="Starmie", moves=("surf",), ability="Natural Cure",
                          item="Leftovers", level=79,
                          evs={s: 85 for s in ("hp", "atk", "def", "spa", "spd", "spe")})


def _maxhp(mon: FixturePokemon, dex: ShowdownDex) -> int:
    info = dex.species_info(mon.species)
    return gen3_hp_stat(int(info.base_stats["hp"]), 31, int((mon.evs or {}).get("hp", 0)), mon.level)


def _override(*, sleeper: FixturePokemon = _SKARMORY, bench: FixturePokemon = _STARMIE) -> BattleStartOverride:
    return BattleStartOverride(
        player_teams={
            "p1": pack_team((_SNORLAX,)),
            "p2": pack_team((sleeper, bench)),
        },
    )


def _payload(
    dex: ShowdownDex,
    *,
    rest_attempts=None,
    refunded_time=None,
    skipped_time=None,
    refund_pending=False,
    sleeper_active=False,
):
    """A minimal two-seat payload whose p2 Skarmory is asleep.

    ``rest_attempts`` is the row annotation under test; ``None`` leaves the row
    unannotated, which is exactly how an opponent-induced sleeper arrives.
    """

    snorlax_hp = _maxhp(_SNORLAX, dex)
    sleeper_row = {
        "species": "Skarmory",
        "condition": "88/100 slp",
        "active": sleeper_active,
    }
    if rest_attempts is not None:
        sleeper_row["restSleepAttempts"] = rest_attempts
    if refunded_time is not None:
        sleeper_row["restSleepRefundedTime"] = refunded_time
    if skipped_time is not None:
        sleeper_row["restSleepSkippedTime"] = skipped_time
    if refund_pending:
        sleeper_row["restSleepActiveRefundPending"] = True
    other_row = {"species": "Starmie", "condition": "100/100", "active": not sleeper_active}
    return {
        "turn": 9,
        "weather": None,
        "weatherSetTurn": None,
        "weatherFromAbility": False,
        "futureSight": {"p1": 0, "p2": 0},
        "wishSetTurns": {},
        "leechSeedSourceSides": {},
        "pendingBatonPassSides": [],
        "deferredOpponentActions": {},
        "deferredOpponentActionPriors": {},
        "selfPlayer": "p1",
        "selfRequestKind": "move",
        "selfTeamOrder": ["Snorlax"],
        "selfActiveRequestState": {
            "trapped": False, "maybeTrapped": False, "maybeDisabled": False, "maybeLocked": False,
        },
        "selfBenchedMoveHistory": False,
        "sides": {
            "p1": {
                "pokemon": [{
                    "species": "Snorlax",
                    "condition": f"{snorlax_hp}/{snorlax_hp}",
                    "active": True,
                    "moves": [{"id": "bodyslam", "pp": 20, "maxpp": 24, "disabled": False}],
                }],
                "boosts": {}, "volatiles": [], "materializationBlockers": [],
                "toxicStage": 0, "sideConditions": {}, "sideConditionSetTurns": {},
            },
            "p2": {
                "pokemon": [sleeper_row, other_row],
                "boosts": {}, "volatiles": [], "materializationBlockers": [],
                "toxicStage": 0, "sideConditions": {}, "sideConditionSetTurns": {},
            },
        },
    }


_LEADS = [
    "|player|p1|Alice|",
    "|player|p2|Bob|",
    "|switch|p1a: Snorlax|Snorlax, L80|100/100",
    "|switch|p2a: Skarmory|Skarmory, L76|100/100",
    "|turn|1",
]


# --- the reconstruction -----------------------------------------------------------

class RestTurnsReconstructionTests(unittest.TestCase):
    """Exact Rest reconstruction from attempts, refunds, and the world ability."""

    def setUp(self) -> None:
        self.dex = _dex()
        probe = _stubbed_capability_probe()
        self.probe_calls = probe.__enter__()
        self.addCleanup(probe.__exit__, None, None, None)

    def _sleeper(self, *, sleeper: FixturePokemon = _SKARMORY, **kwargs):
        world = battle_spec_from_payload(
            _payload(self.dex, **kwargs), _override(sleeper=sleeper), dex=self.dex
        )
        return world.spec.side_two.pokemon[0]

    def test_building_a_rest_sleeper_gates_on_the_capability(self) -> None:
        # The constructor-side half of the gate. An unannotated sleeper must NOT
        # reach for the native probe -- otherwise every ordinary sleep would start
        # requiring a wheel it has no need of.
        self._sleeper(rest_attempts=0)
        self.assertEqual(len(self.probe_calls), 1)

        self.probe_calls.clear()
        battle_spec_from_payload(
            _payload(self.dex), _override(), dex=self.dex, approximate_sleep_turns=True
        )
        self.assertEqual(self.probe_calls, [])

    def test_each_attempt_count_maps_onto_the_engines_own_counter(self) -> None:
        # The engine sets rest_turns 3 on Rest and decrements once per move ATTEMPT,
        # waking at 1. In the ordinary (non-Early-Bird, no-refund) path, k public
        # |cant| lines leave 3 - k -- 3, 2 or 1, never 0.
        for attempts, expected in ((0, 3), (1, 2), (2, 1)):
            with self.subTest(attempts=attempts):
                sleeper = self._sleeper(rest_attempts=attempts)
                self.assertEqual(sleeper.status, "sleep")
                self.assertEqual(sleeper.rest_turns, expected)

    def test_the_conversion_is_three_minus_k_and_not_four_minus_k(self) -> None:
        """The wake-convention pin. See ``engine_world._rest_turns_from_row``.

        Both engines decrement on the attempt and both wake on the attempt whose
        PRE-decrement counter is 1, so their counters are the SAME number at the same
        moment -- there is no +1 offset to correct for. Reading Showdown's counter
        after its decrement against the engine's before its own is what makes ``4 - k``
        look right.

        Asserted as an explicit table because ``4 - k`` gets EVERY row wrong, k=0
        included, and NOTHING CLAMPS IT back into range -- not the range check in
        ``_rest_turns_from_row`` (which guards the INPUT k) and not the adapter (which
        validates only non-negativity). The rows differ only in how the wrongness would
        surface: k=0 builds an unrepresentable 4 that the engine panics on, k=1 and k=2
        build legal-but-late counters.

        So a k=0 check would have caught ``4 - k`` loudly. Every row is pinned for
        REACHABILITY instead: a fresh, un-attempted Rest dominates the corpus, so an
        error confined to k=1/k=2 is the one that could ride along unexercised.
        """

        three_minus_k = {0: 3, 1: 2, 2: 1}
        four_minus_k = {k: 4 - k for k in three_minus_k}

        for attempts, expected in three_minus_k.items():
            with self.subTest(k=attempts):
                self.assertEqual(self._sleeper(rest_attempts=attempts).rest_turns, expected)

        # No row agrees with 4 - k, k=0 included -- the claim the rationale above rests
        # on. If this ever stops holding, the mapping has been rewritten.
        for attempts, wrong in four_minus_k.items():
            with self.subTest(k=attempts):
                self.assertNotEqual(self._sleeper(rest_attempts=attempts).rest_turns, wrong)

        # And k=0's 4 is not merely wrong, it is outside the range the engine can
        # represent at all (gen3 matches 0/1/2/3 and panics on anything else), which is
        # why that row would have failed loudly rather than silently.
        self.assertEqual(four_minus_k[0], 4)
        self.assertNotIn(four_minus_k[0], {0, 1, 2, 3})

    def test_a_fresh_rest_builds_the_full_counter_and_nothing_is_clamped(self) -> None:
        # The k=0 edge. 3 - 0 is 3, exactly what Rest sets -- arrived at directly, not
        # by clamping: the conversion range-checks its INPUT and returns None outside
        # 0..2, so no output is ever squeezed into range.
        self.assertEqual(self._sleeper(rest_attempts=0).rest_turns, 3)
        built = {self._sleeper(rest_attempts=k).rest_turns for k in (0, 1, 2)}
        self.assertEqual(built, {1, 2, 3})

    def test_a_rest_sleep_carries_no_elapsed_turn_count(self) -> None:
        # The engine branches on rest_turns first and reads sleep_turns only in its
        # 0 arm, so a Rest sleep has no elapsed-turn count to carry. Setting one
        # would be read by nothing and would misdescribe the state on inspection.
        self.assertEqual(self._sleeper(rest_attempts=1).sleep_turns, 0)

    def test_a_rest_sleep_needs_no_approximation_flag(self) -> None:
        # The whole point: this is EXACT, so it constructs where the position used
        # to be declined outright.
        self.assertEqual(self._sleeper(rest_attempts=0).rest_turns, 3)

    def test_early_bird_consumes_two_units_per_attempt_and_refunds_one_per_sleep_talk(self) -> None:
        # Rest begins at 3. Early Bird's first sleep attempt leaves 1; its public
        # skippedTime refund restores one when the mon returns, so the constructed
        # world must start at 2 rather than treating it as a fresh three-turn Rest.
        sleeper = self._sleeper(
            sleeper=_SKARMORY_EARLY_BIRD,
            rest_attempts=1,
            skipped_time=1,
        )
        self.assertEqual(sleeper.ability, "earlybird")
        self.assertEqual(sleeper.rest_turns, 2)

    def test_early_bird_keeps_a_refund_after_the_sleeper_has_reentered(self) -> None:
        sleeper = self._sleeper(
            sleeper=_SKARMORY_EARLY_BIRD,
            rest_attempts=1,
            refunded_time=1,
        )
        self.assertEqual(sleeper.rest_turns, 2)

    def test_early_bird_inconsistent_attempts_fail_closed_even_with_approximation(self) -> None:
        # Early Bird's second attempt wakes the mon, so a sleeping public row cannot
        # legitimately carry either count. Exact Rest provenance must not degrade into
        # generic induced sleep just because approximation is enabled.
        for attempts in (2, 3):
            with self.subTest(attempts=attempts):
                with self.assertRaises(EngineWorldUnsupported) as caught:
                    battle_spec_from_payload(
                        _payload(self.dex, rest_attempts=attempts),
                        _override(sleeper=_SKARMORY_EARLY_BIRD),
                        dex=self.dex,
                        approximate_sleep_turns=True,
                    )
                self.assertEqual(caught.exception.reason, "rest_sleep_provenance_unrepresentable")

    def test_repeated_normal_sleep_talk_refunds_allow_a_raw_attempt_count_above_two(self) -> None:
        sleeper = self._sleeper(rest_attempts=3, refunded_time=3)
        self.assertEqual(sleeper.rest_turns, 3)

    def test_pending_refund_now_builds_and_carries_the_bank_unfolded(self) -> None:
        """Producer B no longer refuses -- the engine has somewhere to put the refund.

        Was: assertRaises(rest_sleep_active_refund_pending). This is the class that
        cost 781 decisions in era 58, 28.7% of all fallback.

        The load-bearing assertion is that the refund is NOT folded into rest_turns.
        Folding is right for a benched mon, whose next attempt is necessarily
        preceded by a switch-in; for an active one it double-counts, because the
        engine credits the bank again on switch-in.
        """

        # The attempt counts are part of the row now. Before this change
        # local_showdown set the flag and `continue`d, so restSleepAttempts was
        # never written for exactly these rows -- which is why deleting the refusal
        # alone would have changed nothing: construction is gated on that key.
        payload = _payload(
            self.dex,
            refund_pending=True,
            sleeper_active=True,
            rest_attempts=1,
            skipped_time=1,
        )
        world = battle_spec_from_payload(
            payload,
            _override(),
            dex=self.dex,
            approximate_sleep_turns=True,
        )
        sleeper = world.spec.side_two.pokemon[0]
        self.assertEqual(sleeper.status, "sleep")
        self.assertEqual(sleeper.rest_sleep_pending_refund, 1)
        # _payload writes one attempt and one skipped turn, so the unfolded counter
        # is 3 - 1 = 2 and the folded one would have been 3. Asserting 2 is what
        # distinguishes the two; asserting "non-zero" would pass either way.
        self.assertEqual(sleeper.rest_turns, 2)

    def test_the_split_reason_codes_reach_world_construction_distinctly(self) -> None:
        # The flags are only useful if they survive into the refusal REASON, which is
        # what the fallback ledger counts. Both must refuse even with approximation on.
        # restSleepActiveRefundPending is deliberately NOT in this list any more: it
        # builds now rather than refusing, and its own test asserts that. The other
        # two still refuse, for different owners -- one is a harness snapshot gap,
        # one is a pre-split corpus row whose producer is unrecoverable.
        for flag, reason in (
            ("restSleepAttemptUnsettled", "rest_sleep_attempt_unsettled"),
            ("restSleepRefundPending", "rest_sleep_refund_pending_unsplit_legacy"),
        ):
            with self.subTest(flag=flag):
                payload = _payload(self.dex, sleeper_active=True)
                payload["sides"]["p2"]["pokemon"][0][flag] = True
                with self.assertRaises(EngineWorldUnsupported) as caught:
                    battle_spec_from_payload(
                        payload, _override(), dex=self.dex, approximate_sleep_turns=True
                    )
                self.assertEqual(caught.exception.reason, reason)

    def test_a_real_annotated_row_reaches_its_PRODUCER_code_not_the_legacy_one(self) -> None:
        """The legacy check must stay LAST, and this is what pins that.

        Every other test in this file sets the flags by hand, one at a time, so none
        of them exercises the thing the dual-write created: a live row carrying BOTH
        its producer flag and the pre-split flag. With the checks in the wrong order
        such a row is swallowed by the legacy branch and reported as un-attributable
        — the split silently 100% undone — and review confirmed the whole suite stays
        green when that happens. So: annotate a real stream, prove the row really does
        carry the legacy key, then build it and demand the PRODUCER code.
        """
        talk = [
            "|cant|p2a: Skarmory|slp",
            "|move|p2a: Skarmory|Sleep Talk|p2a: Skarmory",
            "|move|p2a: Skarmory|Splash|p2a: Skarmory|[from]move: Sleep Talk",
            "|upkeep", "|turn|2",
        ]
        # Producer B's expectation is None: it BUILDS now instead of refusing. The
        # ordering guard still holds, and is arguably sharper this way -- if the
        # legacy check were moved ahead of the producer checks, a B row would refuse
        # with the legacy code instead of building, so "builds" is a strictly
        # stronger statement than "refuses with B's code" was.
        cases = (
            ("A", ["|cant|p2a: Skarmory|slp"], "rest_sleep_attempt_unsettled"),
            ("B", talk, None),
        )
        for producer, tail, expected in cases:
            with self.subTest(producer=producer):
                payload = _payload(self.dex, sleeper_active=True)
                rows = payload["sides"]["p2"]["pokemon"]
                lines = [
                    "|player|p1|Alice|", "|player|p2|Bob|",
                    "|switch|p1a: Snorlax|Snorlax, L80|100/100",
                    "|switch|p2a: Skarmory|Skarmory, L76|100/100",
                    "|turn|1",
                    "|move|p2a: Skarmory|Rest|p2a: Skarmory",
                    "|-status|p2a: Skarmory|slp|[from] move: Rest",
                    *tail,
                ]
                _apply_rest_sleep_provenance(
                    rows, parse_showdown_replay(lines, battle_id="order-guard"), "p2"
                )
                # The premise: this row carries BOTH keys, which is what makes the
                # ordering load-bearing in the first place.
                self.assertTrue(rows[0].get("restSleepRefundPending"), rows[0])

                if expected is None:
                    world = battle_spec_from_payload(
                        payload, _override(), dex=self.dex, approximate_sleep_turns=True
                    )
                    sleeper = world.spec.side_two.pokemon[0]
                    self.assertEqual(sleeper.status, "sleep")
                    # The bank is what proves the PRODUCER branch ran rather than a
                    # generic sleep path: only that branch sets it.
                    self.assertEqual(sleeper.rest_sleep_pending_refund, 1)
                else:
                    with self.assertRaises(EngineWorldUnsupported) as caught:
                        battle_spec_from_payload(
                            payload, _override(), dex=self.dex, approximate_sleep_turns=True
                        )
                    self.assertEqual(caught.exception.reason, expected)

    def test_the_retired_reason_code_is_gone_from_the_engine_world_source(self) -> None:
        # A grep-level guard, deliberately. The point of retiring the old name rather
        # than reusing it for producer B is that no historical count can be silently
        # read as one producer's share; that only holds if the name cannot come back.
        source = pathlib.Path(engine_world.__file__).read_text(encoding="utf-8")
        # The QUOTED form only: the prose comment above the split names the retired
        # code on purpose, and a bare substring check would forbid documenting it.
        self.assertNotIn('"rest_sleep_skipped_time_pending"', source)
        self.assertNotIn("'rest_sleep_skipped_time_pending'", source)

    def test_a_valid_unannotated_sleeper_can_still_use_approximation(self) -> None:
        # An opponent-induced sleeper is never annotated, so it must still fail
        # closed without the flag and still build as freshly-asleep with it. This
        # is the regression guard on the arm the fix does NOT change.
        payload = _payload(self.dex)
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertEqual(caught.exception.reason, "status_unsupported")

        world = battle_spec_from_payload(
            payload, _override(), dex=self.dex, approximate_sleep_turns=True
        )
        induced = world.spec.side_two.pokemon[0]
        self.assertEqual(induced.status, "sleep")
        self.assertEqual(induced.rest_turns, 0)
        self.assertEqual(induced.sleep_turns, 0)

    def test_unrepresentable_rest_marker_refuses_approximation(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p2"]["pokemon"][0]["restSleepProvenanceUnrepresentable"] = True
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(
                payload,
                _override(),
                dex=self.dex,
                approximate_sleep_turns=True,
            )
        self.assertEqual(caught.exception.reason, "rest_sleep_provenance_unrepresentable")

    def test_the_exemption_rides_the_bench(self) -> None:
        # The population this exists to serve. An ACTIVE Rest-sleeper reveals itself
        # next turn either way; a BENCHED one is visible to nothing but the clause.
        benched = self._sleeper(rest_attempts=1, sleeper_active=False)
        active = self._sleeper(rest_attempts=1, sleeper_active=True)
        self.assertEqual(benched.rest_turns, 2)
        self.assertEqual(active.rest_turns, 2)

    def test_out_of_range_rest_provenance_fails_closed_even_with_approximation(self) -> None:
        # The raw count can grow beyond two when Sleep Talk is repeatedly refunded
        # across switches. It still must describe a live Rest counter in 1..3.
        for bad in (3, 7, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(EngineWorldUnsupported) as caught:
                    battle_spec_from_payload(
                        _payload(self.dex, rest_attempts=bad),
                        _override(),
                        dex=self.dex,
                        approximate_sleep_turns=True,
                    )
                self.assertEqual(caught.exception.reason, "rest_sleep_provenance_unrepresentable")

    def test_malformed_or_non_integer_rest_provenance_fails_closed_with_approximation(self) -> None:
        # Bools are ints in Python: True must not read as "one attempt spent". Set the
        # key directly so None remains explicit provenance rather than unannotated sleep.
        for bad in (True, False, "1", 1.0, None):
            with self.subTest(bad=bad):
                payload = _payload(self.dex, rest_attempts=0)
                payload["sides"]["p2"]["pokemon"][0]["restSleepAttempts"] = bad
                with self.assertRaises(EngineWorldUnsupported) as caught:
                    battle_spec_from_payload(
                        payload,
                        _override(),
                        dex=self.dex,
                        approximate_sleep_turns=True,
                    )
                self.assertEqual(caught.exception.reason, "rest_sleep_provenance_unrepresentable")

    def test_a_fainted_row_never_carries_a_rest_state(self) -> None:
        payload = _payload(self.dex, rest_attempts=1)
        payload["sides"]["p2"]["pokemon"][0]["condition"] = "0 fnt"
        payload["sides"]["p2"]["pokemon"][1]["active"] = True
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        fainted = world.spec.side_two.pokemon[0]
        self.assertEqual(fainted.hp, 0)
        self.assertEqual(fainted.status, "none")
        self.assertEqual(fainted.rest_turns, 0)


# --- the row annotation -----------------------------------------------------------

class RestSleepRowAnnotationTests(unittest.TestCase):
    """``local_showdown._apply_rest_sleep_provenance``: which rows get the count."""

    @staticmethod
    def _rows():
        return [
            {"species": "Skarmory", "condition": "88/100 slp", "active": False},
            {"species": "Starmie", "condition": "100/100", "active": True},
        ]

    @staticmethod
    def _annotate(lines, player="p2"):
        replay = parse_showdown_replay(lines, battle_id="rest-build")
        rows = RestSleepRowAnnotationTests._rows()
        _apply_rest_sleep_provenance(rows, replay, player)
        return rows

    _RESTED = _LEADS + [
        "|move|p2a: Skarmory|Rest|p2a: Skarmory",
        "|-status|p2a: Skarmory|slp|[from] move: Rest",
    ]

    def test_a_rest_sleeper_is_annotated_with_its_attempt_count(self) -> None:
        rows = self._annotate(self._RESTED)
        self.assertEqual(rows[0]["restSleepAttempts"], 0)

    def test_the_count_advances_with_the_public_cant_lines(self) -> None:
        rows = self._annotate(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|cant|p2a: Skarmory|slp",
            "|upkeep",
        ])
        self.assertEqual(rows[0]["restSleepAttempts"], 2)

    def test_bench_turns_do_not_advance_it(self) -> None:
        # The reason the tracker counts attempts: gen3's timer ticks only in
        # slp.onBeforeMove, so a benched Rest does not advance at all. Elapsed
        # turns would build this mon as closer to waking than it is.
        rows = self._annotate(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|switch|p2a: Starmie|Starmie, L79|100/100",
            "|turn|2",
            "|move|p1a: Snorlax|Body Slam|p2a: Starmie",
            "|turn|3",
            "|move|p1a: Snorlax|Body Slam|p2a: Starmie",
            "|turn|4",
        ])
        self.assertEqual(rows[0]["restSleepAttempts"], 1)

    def test_a_sleep_talk_bench_row_refunds_its_rest_clock(self) -> None:
        """A benched Sleep Talker carries the exact post-refund Rest counter.

        This is the fail-unpatched/pass-patched pin. The old tracker retired its
        Rest provenance at the direct Sleep Talk line, so the now-benched row
        reached world construction as generic sleep. The direct world has no
        ``skippedTime`` field, but a benched mon will receive the refund before
        its next attempt. Preserve both public quantities so Early Bird can apply
        its two-units-per-attempt rule during construction.
        """
        rows = self._annotate(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|move|p2a: Skarmory|Sleep Talk|p2a: Skarmory",
            "|move|p2a: Skarmory|Splash|p2a: Skarmory|[from]move: Sleep Talk",
            "|switch|p2a: Starmie|Starmie, L79|100/100",
            "|upkeep",
            "|turn|2",
        ])
        self.assertEqual(rows[0]["restSleepAttempts"], 1)
        self.assertEqual(rows[0]["restSleepSkippedTime"], 1)

    def test_snore_bench_row_refunds_its_rest_clock_too(self) -> None:
        rows = self._annotate(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|move|p2a: Skarmory|Snore|p1a: Snorlax",
            "|switch|p2a: Starmie|Starmie, L79|100/100",
        ])
        self.assertEqual(rows[0]["restSleepAttempts"], 1)
        self.assertEqual(rows[0]["restSleepSkippedTime"], 1)

    _ACTIVE_SLEEP_TALK = (
        "|cant|p2a: Skarmory|slp",
        "|move|p2a: Skarmory|Sleep Talk|p2a: Skarmory",
        "|move|p2a: Skarmory|Splash|p2a: Skarmory|[from]move: Sleep Talk",
    )

    def _annotate_either(self, lines, *, active):
        """``_annotate``, but with the sleeper's active flag under the test's control."""
        replay = parse_showdown_replay(lines, battle_id="active-sleep-talk")
        rows = self._rows()
        rows[0]["active"] = active
        rows[1]["active"] = not active
        _apply_rest_sleep_provenance(rows, replay, "p2")
        return rows

    def _annotate_active_sleeper(self, lines):
        return self._annotate_either(lines, active=True)

    def test_active_sleep_talk_row_now_carries_its_attempt_counts(self) -> None:
        """Was: assertNotIn("restSleepAttempts").

        Withholding the attempt counts was the bail-out that made producer B a
        refusal, and it is the reason deleting the decline in engine_world would
        have changed nothing on its own -- construction is gated on this key. The
        flag still gets written, and still means "pending, not folded", so a
        checkout predating the engine field keeps refusing rather than reading the
        counts and silently dropping the refund.
        """

        rows = self._annotate_active_sleeper(self._RESTED + [*self._ACTIVE_SLEEP_TALK])
        self.assertEqual(rows[0]["restSleepAttempts"], 1)
        self.assertEqual(rows[0]["restSleepSkippedTime"], 1)
        # Producer B: the Sleep Talk |move| classifies the attempt immediately, so this
        # is the known-value case, not an unsettled one.
        self.assertTrue(rows[0]["restSleepActiveRefundPending"])
        self.assertNotIn("restSleepAttemptUnsettled", rows[0])

    def test_the_two_producers_separate_on_different_axes(self) -> None:
        """The whole 2x2 the split exists to make countable.

        These two were one reason code, and the natural guess -- that they are the
        same situation seen before and after ``|upkeep|`` -- is WRONG. They separate
        on different axes, which is exactly why conflating them made the class
        unsizeable:

        * Producer A fires when the attempt is still UNCLASSIFIED (no ``|upkeep|``
          yet). It does not care whether the sleeper is active, and appending
          ``|upkeep|`` builds the world outright -- so it is a harness/observation
          boundary, not an engine limitation.
        * Producer B fires when the attempt IS classified as a sleep-usable skip and
          the sleeper is ACTIVE. Benched, the same stream builds with the refund
          folded in. Only this one is closed by a pending-skipped-time field on
          ``Pokemon``.

        B is driven by much more than Sleep Talk, and NOT by an enumerable little
        list. Three routes reach `_mark_pending_rest_sleep_refundable`:

        * a sleep-usable ``|move|`` -- Sleep Talk or Snore;
        * ``|-activate|`` filtered to ``confusion`` / ``moveattract``;
        * **any later same-actor** ``|cant|`` -- that branch (``showdown.py:3281``)
          has **no reason filter at all**. Its own comment names flinch/Truant/
          paralysis/Attract; `recharge` and `nomoves` reach it too and are named by
          nothing. So the rule is what this test asserts, not a list. Two earlier
          versions of this docstring got that wrong in opposite ways: one claimed
          "six line families" (an undercount read off a comment rather than the
          code), the next implied `par` was unnamed when the comment names it.
        """
        upkeep = ["|upkeep", "|turn|2"]
        bare = ["|cant|p2a: Skarmory|slp"]

        # --- producer A: unclassified, either seat position, and it is recoverable.
        for active in (False, True):
            with self.subTest(producer="A", active=active):
                rows = self._annotate_either(self._RESTED + bare, active=active)
                self.assertTrue(rows[0]["restSleepAttemptUnsettled"])
                self.assertNotIn("restSleepActiveRefundPending", rows[0])
                # The proof it is not an engine gap: settle the turn and it builds.
                settled = self._annotate_either(self._RESTED + bare + upkeep, active=active)
                self.assertEqual(settled[0]["restSleepAttempts"], 1)
                self.assertNotIn("restSleepAttemptUnsettled", settled[0])

        # --- producer B: classified skip; refused ONLY while active. Sampled across
        #     all three routes, including `cant` reasons that no comment enumerates.
        skip_families = {
            "move:sleeptalk": list(self._ACTIVE_SLEEP_TALK[1:]),
            "move:snore": ["|move|p2a: Skarmory|Snore|p1a: Snorlax"],
            "cant:flinch": ["|cant|p2a: Skarmory|flinch"],
            "cant:truant": ["|cant|p2a: Skarmory|ability: Truant"],
            # `-activate` route ONLY -- no trailing `|cant|`. With the `|cant|` line
            # present these two pass via the unfiltered `cant` route instead, so
            # deleting "moveattract" from the `-activate` filter was a SILENT
            # mutation. Review caught that; the filter had zero coverage repo-wide.
            "activate:confusion": ["|-activate|p2a: Skarmory|confusion",
                                   "|-damage|p2a: Skarmory|80/100"],
            "activate:attract": ["|-activate|p2a: Skarmory|move: Attract"],
        }
        # Code-property probes, NOT claimed producers. Gen 3 major statuses are
        # exclusive, so a `slp` row cannot emit `|cant|...|par`, and a mon cannot
        # self-Rest while recharging. They are here to pin that the `cant` branch is
        # unfiltered -- a property of THIS annotator, characterised, not validated
        # against the Node sim this file names as ground truth. If gen 3 does not in
        # fact refund these, that is a bug these subtests would protect.
        unfiltered_probes = {
            "cant:par": ["|cant|p2a: Skarmory|par"],
            "cant:recharge": ["|cant|p2a: Skarmory|recharge"],
            "cant:nomoves": ["|cant|p2a: Skarmory|nomoves"],
        }
        for name, events in {**skip_families, **unfiltered_probes}.items():
            with self.subTest(producer="B", family=name):
                lines = self._RESTED + bare + events + upkeep

                benched = self._annotate_either(lines, active=False)
                self.assertEqual(benched[0]["restSleepAttempts"], 1)
                self.assertEqual(benched[0]["restSleepSkippedTime"], 1)
                self.assertNotIn("restSleepActiveRefundPending", benched[0])

                active_row = self._annotate_either(lines, active=True)
                self.assertTrue(active_row[0]["restSleepActiveRefundPending"])
                self.assertNotIn("restSleepAttemptUnsettled", active_row[0])

    def test_the_pre_split_flag_is_never_emitted_alone(self) -> None:
        """A live row sets the old flag TOO, but never on its own.

        Writing it keeps a pre-split checkout refusing rather than silently
        approximating (`_mark_legacy_rest_refund_pending`). Writing it *alone* is
        what must never happen: the third legacy reason code is defined as "old flag,
        neither producer flag", so a live row emitting it bare would be counted as
        un-attributable and the split would lose exactly the traffic it exists to
        separate.
        """
        streams = [
            [],
            ["|cant|p2a: Skarmory|slp"],
            [*self._ACTIVE_SLEEP_TALK],
            [*self._ACTIVE_SLEEP_TALK, "|upkeep", "|turn|2"],
            ["|cant|p2a: Skarmory|slp", "|move|p2a: Skarmory|Snore|p1a: Snorlax",
             "|upkeep", "|turn|2"],
            ["|cant|p2a: Skarmory|slp", "|cant|p2a: Skarmory|flinch",
             "|upkeep", "|turn|2"],
            ["|cant|p2a: Skarmory|slp", "|switch|p2a: Starmie|Starmie, L79|100/100",
             "|upkeep", "|turn|2"],
        ]
        # Every OTHER branch of the annotator too, not just the refusing pair. Review
        # showed the first version of this guard missed a stray bare write in the
        # `induced` branch entirely, because none of the streams above reach it: each
        # of these ends at a different `continue`.
        other_branches = [
            # induced sleep -> restSleepProvenanceUnrepresentable
            (_LEADS + ["|move|p1a: Snorlax|Hypnosis|p2a: Skarmory",
                       "|-status|p2a: Skarmory|slp"], False),
            # own Rest, then induced by the opponent as well -> the `induced` conflict
            (self._RESTED + ["|move|p1a: Snorlax|Hypnosis|p2a: Skarmory",
                             "|-status|p2a: Skarmory|slp"], False),
            # woken -> annotation retired
            (self._RESTED + ["|-curestatus|p2a: Skarmory|slp"], False),
        ]

        producer_flags = ("restSleepAttemptUnsettled", "restSleepActiveRefundPending")
        cases = [(self._RESTED + t, a) for t in streams for a in (False, True)]
        cases += other_branches
        for index, (lines, active) in enumerate(cases):
            with self.subTest(case=index, active=active):
                for row in self._annotate_either(lines, active=active):
                    if "restSleepRefundPending" not in row:
                        continue
                    self.assertTrue(
                        any(flag in row for flag in producer_flags),
                        f"row emitted the pre-split flag alone: {row}",
                    )

    def test_a_live_row_still_refuses_under_a_pre_split_consumer(self) -> None:
        """The mirror hazard the flag rename would otherwise create.

        A pre-split `engine_world` knows neither producer flag. If a live row carried
        only those, it would match no branch there, fall through to
        ``approximate_sleep_turns`` and build ``rest_turns=0`` -- a silently wrong
        world, strictly worse than the mislabelled refusal the rename fixed. Simulated
        by asking only the question a pre-split consumer can ask.
        """
        cases = (
            ["|cant|p2a: Skarmory|slp"],                                  # producer A
            [*self._ACTIVE_SLEEP_TALK, "|upkeep", "|turn|2"],             # producer B
        )
        for index, tail in enumerate(cases):
            for active in (False, True):
                with self.subTest(case=index, active=active):
                    rows = self._annotate_either(self._RESTED + tail, active=active)
                    row = rows[0]
                    if not any(f in row for f in
                               ("restSleepAttemptUnsettled", "restSleepActiveRefundPending")):
                        continue  # this combination builds; nothing to preserve
                    self.assertTrue(
                        row.get("restSleepRefundPending"),
                        "a refusing live row must still carry the pre-split flag, or a "
                        "pre-split consumer will approximate it instead of refusing",
                    )


    def test_interrupted_sleep_usable_attempts_keep_the_switch_refund(self) -> None:
        # The sleep handler (priority 10) increments skippedTime before each of
        # these lower-priority gates. No direct Sleep Talk/Snore |move| follows
        # when the gate aborts the action, but the next switch still refunds it.
        interruptions = (
            ("flinch", ("|cant|p2a: Skarmory|flinch",)),
            ("truant", ("|cant|p2a: Skarmory|ability: Truant",)),
            ("confusion", ("|-activate|p2a: Skarmory|confusion", "|-damage|p2a: Skarmory|80/100")),
            ("attract", ("|-activate|p2a: Skarmory|move: Attract", "|cant|p2a: Skarmory|Attract")),
        )
        for name, events in interruptions:
            with self.subTest(interruption=name):
                rows = self._annotate(self._RESTED + [
                    "|cant|p2a: Skarmory|slp",
                    *events,
                    "|switch|p2a: Starmie|Starmie, L79|100/100",
                    "|upkeep",
                    "|turn|2",
                ])
                self.assertEqual(rows[0]["restSleepAttempts"], 1)
                self.assertEqual(rows[0]["restSleepSkippedTime"], 1)

    def test_unresolved_sleep_attempt_is_explicitly_refused(self) -> None:
        rows = self._annotate(self._RESTED + ["|cant|p2a: Skarmory|slp"])
        self.assertNotIn("restSleepAttempts", rows[0])
        # Producer A, since the attempt is still unclassified at the snapshot.
        self.assertTrue(rows[0]["restSleepAttemptUnsettled"])

    def test_benched_sleep_talk_clock_does_not_tick_until_switch_back(self) -> None:
        rows = self._annotate(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|move|p2a: Skarmory|Sleep Talk|p2a: Skarmory",
            "|move|p2a: Skarmory|Splash|p2a: Skarmory|[from]move: Sleep Talk",
            "|switch|p2a: Starmie|Starmie, L79|100/100",
            "|upkeep",
            "|turn|2",
            "|move|p1a: Snorlax|Body Slam|p2a: Starmie",
            "|upkeep",
            "|turn|3",
            "|move|p1a: Snorlax|Body Slam|p2a: Starmie",
            "|upkeep",
            "|turn|4",
        ])
        self.assertEqual(rows[0]["restSleepAttempts"], 1)
        self.assertEqual(rows[0]["restSleepSkippedTime"], 1)

    def test_switch_back_applies_the_skipped_time_refund_once(self) -> None:
        lines = self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|move|p2a: Skarmory|Sleep Talk|p2a: Skarmory",
            "|move|p2a: Skarmory|Splash|p2a: Skarmory|[from]move: Sleep Talk",
            "|switch|p2a: Starmie|Starmie, L79|100/100",
            "|upkeep",
            "|turn|2",
            "|switch|p2a: Skarmory|Skarmory, L76|88/100 slp",
        ]
        replay = parse_showdown_replay(lines, battle_id="sleep-talk-return")
        rows = self._rows()
        rows[0]["active"] = True
        rows[1]["active"] = False
        _apply_rest_sleep_provenance(rows, replay, "p2")
        self.assertEqual(rows[0]["restSleepAttempts"], 1)
        self.assertEqual(rows[0]["restSleepRefundedTime"], 1)
        self.assertEqual(dict(replay.rest_sleep_skipped_turns), {})
        self.assertEqual(dict(replay.rest_sleep_refunded_turns), {"p2:skarmory": 1})

    def test_two_public_sleep_talk_pivots_accumulate_refunds_and_rebuild_exactly(self) -> None:
        lines = self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|move|p2a: Skarmory|Sleep Talk|p2a: Skarmory",
            "|move|p2a: Skarmory|Splash|p2a: Skarmory|[from]move: Sleep Talk",
            "|switch|p2a: Starmie|Starmie, L79|100/100",
            "|upkeep",
            "|turn|2",
            "|switch|p2a: Skarmory|Skarmory, L76|88/100 slp",
            "|cant|p2a: Skarmory|slp",
            "|move|p2a: Skarmory|Sleep Talk|p2a: Skarmory",
            "|move|p2a: Skarmory|Splash|p2a: Skarmory|[from]move: Sleep Talk",
            "|switch|p2a: Starmie|Starmie, L79|100/100",
            "|upkeep",
            "|turn|3",
            "|switch|p2a: Skarmory|Skarmory, L76|88/100 slp",
        ]
        replay = parse_showdown_replay(lines, battle_id="two-sleep-talk-refunds")
        self.assertEqual(dict(replay.rest_sleep_counts), {"p2:skarmory": 2})
        self.assertEqual(dict(replay.rest_sleep_refunded_turns), {"p2:skarmory": 2})
        self.assertEqual(dict(replay.rest_sleep_skipped_turns), {})

        rows = self._rows()
        rows[0]["active"] = True
        rows[1]["active"] = False
        _apply_rest_sleep_provenance(rows, replay, "p2")
        self.assertEqual(rows[0]["restSleepAttempts"], 2)
        self.assertEqual(rows[0]["restSleepRefundedTime"], 2)

        payload = _payload(_dex())
        payload["sides"]["p2"]["pokemon"] = rows
        with _stubbed_capability_probe():
            world = battle_spec_from_payload(payload, _override(), dex=_dex())
        self.assertEqual(world.spec.side_two.pokemon[0].rest_turns, 3)

    def test_plain_sleep_turn_after_sleep_talk_cancels_the_refund(self) -> None:
        rows = self._annotate(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|move|p2a: Skarmory|Sleep Talk|p2a: Skarmory",
            "|move|p2a: Skarmory|Splash|p2a: Skarmory|[from]move: Sleep Talk",
            "|upkeep",
            "|turn|2",
            "|cant|p2a: Skarmory|slp",
            "|move|p1a: Snorlax|Body Slam|p2a: Skarmory",
            "|upkeep",
            "|turn|3",
            "|switch|p2a: Starmie|Starmie, L79|100/100",
        ])
        self.assertEqual(rows[0]["restSleepAttempts"], 2)

    def test_an_awake_sleep_talk_user_has_nothing_to_retire(self) -> None:
        """Case A: Sleep Talk with no Rest entry in play must be inert.

        Retirement is written as an unconditional ``pop`` on the actor's key, so today
        this holds by entry LIFETIME rather than by any awake/asleep test: no Rest, no
        entry, nothing to remove. That derivation is exactly what makes it worth pinning
        -- a later change that let ``rest_sleep_counts`` outlive the sleep it describes
        would turn every awake Sleep Talk into a spurious retire, and nothing would fail.
        """
        replay = parse_showdown_replay(_LEADS + [
            "|move|p2a: Skarmory|Sleep Talk|p2a: Skarmory",
            "|upkeep",
            "|turn|2",
        ], battle_id="awake-talk")
        self.assertEqual(dict(replay.rest_sleep_counts), {})

    def test_a_sleep_talk_after_waking_does_not_retire_a_later_rest(self) -> None:
        """Case E: Rest -> wake -> awake Sleep Talk, then a SECOND Rest.

        The wake already retired the first entry; the awake Sleep Talk that follows must
        not reach forward and disturb the fresh Rest that comes after it. Pins that
        retirement is scoped to the sleep it belongs to and not to the Pokemon.
        """
        rows = self._annotate(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|-curestatus|p2a: Skarmory|slp",          # first Rest ends
            "|move|p2a: Skarmory|Sleep Talk|p2a: Skarmory",  # awake now: inert
            "|upkeep",
            "|turn|2",
            "|move|p2a: Skarmory|Rest|p2a: Skarmory",  # a brand-new Rest
            "|-status|p2a: Skarmory|slp|[from] move: Rest",
            "|cant|p2a: Skarmory|slp",
            "|upkeep",
        ])
        self.assertEqual(rows[0]["restSleepAttempts"], 1)

    def test_an_ordinary_sleeping_turn_keeps_the_clock(self) -> None:
        # The control: only sleepUsable moves accrue skippedTime, so an ordinary Rest
        # is untouched by this rule and stays exactly gated.
        rows = self._annotate(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|move|p1a: Snorlax|Body Slam|p2a: Skarmory",
            "|upkeep",
            "|turn|2",
        ])
        self.assertEqual(rows[0]["restSleepAttempts"], 1)

    def test_the_other_seats_sleep_talk_does_not_retire_this_ones_clock(self) -> None:
        rows = self._annotate(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax",
            "|upkeep",
        ])
        self.assertEqual(rows[0]["restSleepAttempts"], 1)

    def test_an_induced_sleeper_is_never_annotated(self) -> None:
        rows = self._annotate(_LEADS + [
            "|move|p1a: Snorlax|Hypnosis|p2a: Skarmory",
            "|-status|p2a: Skarmory|slp",
        ])
        self.assertNotIn("restSleepAttempts", rows[0])

    def test_waking_retires_the_annotation(self) -> None:
        rows = self._annotate(self._RESTED + ["|-curestatus|p2a: Skarmory|slp"])
        self.assertNotIn("restSleepAttempts", rows[0])

    def test_only_the_sleeping_mons_row_is_touched(self) -> None:
        rows = self._annotate(self._RESTED)
        self.assertNotIn("restSleepAttempts", rows[1])

    def test_the_other_seats_rows_are_never_annotated(self) -> None:
        # Keys are side-scoped, so p2's Rest must not bleed onto a p1 row that
        # happens to carry the same species name.
        replay = parse_showdown_replay(self._RESTED, battle_id="rest-build")
        rows = [{"species": "Skarmory", "condition": "88/100 slp", "active": True}]
        _apply_rest_sleep_provenance(rows, replay, "p1")
        self.assertNotIn("restSleepAttempts", rows[0])

    def test_a_baton_pass_does_not_disturb_a_benched_rest_sleeper(self) -> None:
        # Composition guard against PR #879, which landed in the same parser file
        # and widened which Baton-Passed volatiles transfer and materialize
        # (perish counters, confusion, partial trap). Sleep is a STATUS, not a
        # volatile: ``copyVolatileFrom`` never moves it, and #879 changed only the
        # two BP allowlists -- none of the ``-curestatus`` / ``-cureteam`` /
        # ``faint`` handlers this tracker's clears ride on. A benched Rest must
        # therefore come through a pass with its provenance and count intact.
        rows = self._annotate(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|switch|p2a: Starmie|Starmie, L79|100/100",
            "|turn|2",
            "|-start|p2a: Starmie|perish3",
            "|move|p2a: Starmie|Baton Pass|p2a: Starmie",
            "|switch|p2a: Starmie|Starmie, L79|100/100|[from] Baton Pass",
            "|-start|p2a: Starmie|perish2",
            "|turn|3",
        ])
        self.assertEqual(rows[0]["restSleepAttempts"], 1)

    def test_a_malformed_row_is_skipped_rather_than_crashing(self) -> None:
        replay = parse_showdown_replay(self._RESTED, battle_id="rest-build")
        rows = [{"condition": "88/100 slp"}, {"species": None, "condition": "1/1"}]
        _apply_rest_sleep_provenance(rows, replay, "p2")
        self.assertEqual(rows, [{"condition": "88/100 slp"}, {"species": None, "condition": "1/1"}])

    def test_malformed_rest_maps_refuse_approximate_sleep_materialization(self) -> None:
        cases = {
            "count": ("rest_sleep_counts", "not-an-int"),
            "refunded": ("rest_sleep_refunded_turns", "not-an-int"),
            "skipped": ("rest_sleep_skipped_turns", "not-an-int"),
            "pending": ("rest_sleep_pending_attempt", "not-a-bool"),
            "inconsistent_refund": ("rest_sleep_refunded_turns", 2),
            "orphan_refund": ("rest_sleep_refunded_turns", 1),
            "orphan_skipped": ("rest_sleep_skipped_turns", 1),
            "orphan_pending": ("rest_sleep_pending_attempt", True),
        }
        for name, (attribute, value) in cases.items():
            with self.subTest(case=name):
                replay = parse_showdown_replay(self._RESTED, battle_id=f"rest-invalid-{name}")
                if name.startswith("orphan_"):
                    replay.rest_sleep_counts.clear()
                getattr(replay, attribute)["p2:skarmory"] = value
                rows = self._rows()
                _apply_rest_sleep_provenance(rows, replay, "p2")

                self.assertTrue(rows[0].get("restSleepProvenanceUnrepresentable"))
                self.assertNotIn("restSleepAttempts", rows[0])

                payload = _payload(_dex())
                payload["sides"]["p2"]["pokemon"] = rows
                with self.assertRaises(EngineWorldUnsupported) as caught:
                    battle_spec_from_payload(
                        payload,
                        _override(),
                        dex=_dex(),
                        approximate_sleep_turns=True,
                )
                self.assertEqual(caught.exception.reason, "rest_sleep_provenance_unrepresentable")

    def test_conflicting_induced_and_rest_provenance_refuses_approximation(self) -> None:
        replay = parse_showdown_replay(self._RESTED, battle_id="rest-induced-conflict")
        replay.induced_sleep_victims["p1"] = ("p2:skarmory",)
        rows = self._rows()
        _apply_rest_sleep_provenance(rows, replay, "p2")

        self.assertTrue(rows[0].get("restSleepProvenanceUnrepresentable"))
        payload = _payload(_dex())
        payload["sides"]["p2"]["pokemon"] = rows
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(
                payload,
                _override(),
                dex=_dex(),
                approximate_sleep_turns=True,
            )
        self.assertEqual(caught.exception.reason, "rest_sleep_provenance_unrepresentable")

    def test_cosmetic_forme_rest_key_rebuilds_instead_of_approximating_sleep(self) -> None:
        lines = [
            "|player|p1|Alice|",
            "|player|p2|Bob|",
            "|switch|p1a: Snorlax|Snorlax, L80|100/100",
            "|switch|p2a: Unown|Unown-Z, L76|100/100",
            "|move|p2a: Unown|Rest|p2a: Unown",
            "|-status|p2a: Unown|slp|[from] move: Rest",
            "|cant|p2a: Unown|slp",
            "|upkeep",
        ]
        replay = parse_showdown_replay(lines, battle_id="unown-rest-forme")
        rows = [
            {"species": "Unown-Z", "condition": "88/100 slp", "active": False},
            {"species": "Starmie", "condition": "100/100", "active": True},
        ]
        _apply_rest_sleep_provenance(rows, replay, "p2")
        self.assertEqual(rows[0]["restSleepAttempts"], 1)
        self.assertNotIn("restSleepProvenanceUnrepresentable", rows[0])

        payload = _payload(_dex())
        payload["sides"]["p2"]["pokemon"] = rows
        with _stubbed_capability_probe():
            world = battle_spec_from_payload(
                payload,
                _override(sleeper=_UNOWN_Z),
                dex=_dex(),
                approximate_sleep_turns=True,
            )
        self.assertEqual(world.spec.side_two.pokemon[0].rest_turns, 2)

    def test_ambiguous_cosmetic_formes_refuse_generic_sleep_approximation(self) -> None:
        lines = [
            "|player|p1|Alice|",
            "|player|p2|Bob|",
            "|switch|p1a: Snorlax|Snorlax, L80|100/100",
            "|switch|p2a: Unown|Unown-Z, L76|100/100",
            "|move|p2a: Unown|Rest|p2a: Unown",
            "|-status|p2a: Unown|slp|[from] move: Rest",
            "|cant|p2a: Unown|slp",
            "|upkeep",
        ]
        replay = parse_showdown_replay(lines, battle_id="unown-ambiguous-formes")
        rows = [
            {"species": "Unown-Z", "condition": "88/100 slp", "active": False},
            {"species": "Unown-Question", "condition": "100/100 slp", "active": True},
        ]
        _apply_rest_sleep_provenance(rows, replay, "p2")
        for row in rows:
            self.assertTrue(row.get("restSleepProvenanceUnrepresentable"))
            self.assertNotIn("restSleepAttempts", row)

        payload = _payload(_dex())
        payload["sides"]["p2"]["pokemon"] = rows
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(
                payload,
                _override(sleeper=_UNOWN_Z, bench=_UNOWN_QUESTION),
                dex=_dex(),
                approximate_sleep_turns=True,
            )
        self.assertEqual(caught.exception.reason, "rest_sleep_provenance_unrepresentable")

    def test_unmatched_rest_tracker_refuses_generic_sleep_approximation(self) -> None:
        replay = parse_showdown_replay(self._RESTED, battle_id="unmatched-rest-tracker")
        rows = [
            {"species": "Starmie", "condition": "88/100 slp", "active": False},
            {"species": "Skarmory", "condition": "100/100", "active": True},
        ]
        _apply_rest_sleep_provenance(rows, replay, "p2")
        self.assertTrue(rows[0].get("restSleepProvenanceUnrepresentable"))

        payload = _payload(_dex())
        payload["sides"]["p2"]["pokemon"] = rows
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(
                payload,
                _override(),
                dex=_dex(),
                approximate_sleep_turns=True,
            )
        self.assertEqual(caught.exception.reason, "rest_sleep_provenance_unrepresentable")

    @classmethod
    def _all_rest_maps_live_lines(cls):
        return cls._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|move|p2a: Skarmory|Sleep Talk|p2a: Skarmory",
            "|move|p2a: Skarmory|Splash|p2a: Skarmory|[from]move: Sleep Talk",
            "|upkeep",
            "|turn|2",
            "|cant|p2a: Skarmory|slp",
        ]

    def _assert_rest_maps_cleared(self, replay) -> None:
        for mapping in (
            replay.rest_sleep_counts,
            replay.rest_sleep_refunded_turns,
            replay.rest_sleep_skipped_turns,
            replay.rest_sleep_pending_attempt,
        ):
            self.assertNotIn("p2:skarmory", mapping)

    def test_snapshot_restores_pending_and_skipped_state(self) -> None:
        parser = _ReplayParser("rest-snapshot")
        parser.feed(self._all_rest_maps_live_lines())
        restored = _ReplayParser.from_snapshot(parser.snapshot())
        self.assertEqual(restored.rest_sleep_counts, {"p2:skarmory": 2})
        self.assertEqual(restored.rest_sleep_refunded_turns, {})
        self.assertEqual(restored.rest_sleep_skipped_turns, {"p2:skarmory": 1})
        self.assertEqual(restored.rest_sleep_pending_attempt, {"p2:skarmory": True})

    def test_snapshot_restores_applied_refund_state(self) -> None:
        parser = _ReplayParser("rest-refund-snapshot")
        parser.feed(self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|move|p2a: Skarmory|Sleep Talk|p2a: Skarmory",
            "|switch|p2a: Starmie|Starmie, L79|100/100",
            "|turn|2",
            "|switch|p2a: Skarmory|Skarmory, L76|88/100 slp",
        ])
        restored = _ReplayParser.from_snapshot(parser.snapshot())
        self.assertEqual(restored.rest_sleep_counts, {"p2:skarmory": 1})
        self.assertEqual(restored.rest_sleep_refunded_turns, {"p2:skarmory": 1})
        self.assertEqual(restored.rest_sleep_skipped_turns, {})
        self.assertEqual(restored.rest_sleep_pending_attempt, {})

    def test_natural_cure_clears_all_rest_maps(self) -> None:
        parser = _ReplayParser("rest-natural-cure")
        parser.feed(self._all_rest_maps_live_lines())
        parser.feed([
            "|switch|p2a: Starmie|Starmie, L79|100/100",
            "|-curestatus|p2a: Skarmory|slp|[silent]",
        ])
        self._assert_rest_maps_cleared(parser)

    def test_team_cure_clears_all_rest_maps(self) -> None:
        parser = _ReplayParser("rest-team-cure")
        parser.feed(self._all_rest_maps_live_lines())
        parser.feed(["|-cureteam|p2a: Starmie"])
        self._assert_rest_maps_cleared(parser)

    def test_faint_clears_all_rest_maps(self) -> None:
        parser = _ReplayParser("rest-faint")
        parser.feed(self._all_rest_maps_live_lines())
        parser.feed(["|faint|p2a: Skarmory"])
        self._assert_rest_maps_cleared(parser)

    def test_cleanup_also_clears_an_applied_refund(self) -> None:
        cleanup_lines = {
            "natural_cure": [
                "|switch|p2a: Starmie|Starmie, L79|100/100",
                "|-curestatus|p2a: Skarmory|slp|[silent]",
            ],
            "team_cure": ["|-cureteam|p2a: Starmie"],
            "faint": ["|faint|p2a: Skarmory"],
        }
        prefix = self._RESTED + [
            "|cant|p2a: Skarmory|slp",
            "|move|p2a: Skarmory|Sleep Talk|p2a: Skarmory",
            "|switch|p2a: Starmie|Starmie, L79|100/100",
            "|turn|2",
            "|switch|p2a: Skarmory|Skarmory, L76|88/100 slp",
        ]
        for name, lines in cleanup_lines.items():
            with self.subTest(cleanup=name):
                parser = _ReplayParser(f"rest-refund-{name}")
                parser.feed(prefix)
                self.assertEqual(parser.rest_sleep_refunded_turns, {"p2:skarmory": 1})
                parser.feed(lines)
                self._assert_rest_maps_cleared(parser)


# --- leakage ----------------------------------------------------------------------

class RestSleepLeakageTests(unittest.TestCase):
    """Every input is public-protocol-derived, so BOTH seats compute the same k."""

    _LINES = _LEADS + [
        "|move|p2a: Skarmory|Rest|p2a: Skarmory",
        "|-status|p2a: Skarmory|slp|[from] move: Rest",
        "|cant|p2a: Skarmory|slp",
        "|switch|p2a: Starmie|Starmie, L79|100/100",
        "|turn|2",
    ]

    def test_the_count_is_a_pure_function_of_the_public_lines(self) -> None:
        # No request, no packed team, no belief sample: the parser is fed protocol
        # lines and nothing else, and that is enough to produce the annotation.
        replay = parse_showdown_replay(self._LINES, battle_id="leak")
        rows = [{"species": "Skarmory", "condition": "88/100 slp", "active": False}]
        _apply_rest_sleep_provenance(rows, replay, "p2")
        self.assertEqual(rows[0]["restSleepAttempts"], 1)

    def test_both_seats_derive_the_same_count_from_the_same_lines(self) -> None:
        # The real leakage claim. Skarmory's Rest is p2's own business, but every
        # line it is derived from -- |move| Rest, |-status| ... [from] move: Rest,
        # |cant| ... slp -- is broadcast to both players, so p1 reconstructs the
        # identical counter. A value only one seat could compute would be hidden
        # state wearing a public field's name.
        p2_view = parse_showdown_replay(self._LINES, battle_id="leak-p2")
        p1_view = parse_showdown_replay(self._LINES, battle_id="leak-p1")
        self.assertEqual(dict(p1_view.rest_sleep_counts), dict(p2_view.rest_sleep_counts))
        self.assertEqual(set(p1_view.rest_sleep_counts), {"p2:skarmory"})

    def test_lines_the_public_never_sees_cannot_contribute(self) -> None:
        # A Rest whose -status line was never broadcast leaves no trace at all:
        # there is no other surface the tracker can read, so nothing can smuggle
        # a count in through it.
        silent = [line for line in self._LINES if "move: Rest" not in line]
        replay = parse_showdown_replay(silent, battle_id="leak-silent")
        self.assertEqual(dict(replay.rest_sleep_counts), {})
        rows = [{"species": "Skarmory", "condition": "88/100 slp", "active": False}]
        _apply_rest_sleep_provenance(rows, replay, "p2")
        self.assertNotIn("restSleepAttempts", rows[0])


# --- the capability gate ----------------------------------------------------------

class RestTurnsCapabilityTests(unittest.TestCase):
    """``require_rest_turns_support``: the move-trap/charge-state house pattern."""

    def test_the_probe_accepts_the_patched_wheel(self) -> None:
        from pokezero.poke_engine_adapter import require_rest_turns_support
        from pokezero.poke_engine_backend import probe_poke_engine

        if not probe_poke_engine().ready:
            self.skipTest(
                "poke-engine is not installed/ready; rebuild with "
                "scripts/setup_poke_engine.sh /path/to/venv/bin/python"
            )
        require_rest_turns_support()

    def test_the_probe_fails_closed_on_an_engine_that_drops_the_counter(self) -> None:
        # rest_turns is an UPSTREAM field, so a binding accepting the keyword proves
        # nothing. A binding that takes it and drops it builds the Rest-sleeper as an
        # ordinary sleeper and re-arms the clause -- silently. Only the round trip
        # separates the two.
        from types import SimpleNamespace

        from pokezero.poke_engine_adapter import (
            PokeEngineRestTurnsUnsupportedError,
            require_rest_turns_support,
        )

        class _DroppingState:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def to_string(self) -> str:
                return "SNORLAX,SLEEP,rest_turns-went-missing"

        engine = SimpleNamespace(
            State=_DroppingState,
            Side=lambda **kwargs: SimpleNamespace(**kwargs),
            Pokemon=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        with self.assertRaises(PokeEngineRestTurnsUnsupportedError) as caught:
            require_rest_turns_support(engine)
        self.assertIn("setup_poke_engine.sh", str(caught.exception))

    def test_the_probe_fails_closed_on_a_half_installed_binding(self) -> None:
        from types import SimpleNamespace

        from pokezero.poke_engine_adapter import (
            PokeEngineRestTurnsUnsupportedError,
            require_rest_turns_support,
        )

        with self.assertRaises(PokeEngineRestTurnsUnsupportedError):
            require_rest_turns_support(SimpleNamespace())

    def test_a_rest_spec_gates_on_the_capability_at_render_time(self) -> None:
        # The adapter-side half of the gate: a spec carrying rest_turns must not be
        # handed to an engine that would drop it, even if the world constructor was
        # bypassed.
        from types import SimpleNamespace

        from pokezero.poke_engine_adapter import (
            MoveSpec,
            PokeEngineRestTurnsUnsupportedError,
            PokemonSpec,
            SideSpec,
            build_poke_engine_state,
        )
        from pokezero.poke_engine_adapter import BattleSpec

        def _mon(rest_turns: int) -> PokemonSpec:
            return PokemonSpec(
                id="snorlax", level=80, types=("normal",), hp=100, maxhp=100,
                attack=100, defense=100, special_attack=100, special_defense=100,
                speed=100, moves=(MoveSpec(id="bodyslam"),), status="sleep",
                rest_turns=rest_turns,
            )

        class _DroppingState:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def to_string(self) -> str:
                return "rest_turns-went-missing"

        engine = SimpleNamespace(
            State=_DroppingState,
            Side=lambda **kwargs: SimpleNamespace(**kwargs),
            Pokemon=lambda **kwargs: SimpleNamespace(**kwargs),
            Move=lambda **kwargs: SimpleNamespace(**kwargs),
            SideConditions=lambda **kwargs: SimpleNamespace(**kwargs),
            VolatileStatusDurations=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        resting = BattleSpec(
            side_one=SideSpec(pokemon=(_mon(2),)),
            side_two=SideSpec(pokemon=(_mon(0),)),
        )
        with self.assertRaises(PokeEngineRestTurnsUnsupportedError) as caught:
            build_poke_engine_state(resting, module=engine)
        self.assertIn("setup_poke_engine.sh", str(caught.exception))

        # The control that makes the assertion above a measurement of rest_turns
        # rather than of the fake engine: the SAME engine renders a spec with no
        # Rest on it. Without this the test would pass on any construction error,
        # since every capability error subclasses PokeEngineUnavailableError.
        awake = BattleSpec(
            side_one=SideSpec(pokemon=(_mon(0),)),
            side_two=SideSpec(pokemon=(_mon(0),)),
        )
        build_poke_engine_state(awake, module=engine)


if __name__ == "__main__":
    unittest.main()
