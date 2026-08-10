"""``last_used_move`` provenance: parser truth table, world seeding, engine consequences.

Why this file exists. The engine has implemented gen3's same-turn Encore redirection all
along (``generate_instructions.rs``, mirroring Showdown's ``encore.condition.onOverrideAction``).
Eleven rows of the cycle-nine census looked like a missing redirect and were not: the world
handed the engine ``LastUsedMove::None`` for the target, so Encore's own ``onStart`` guard —
Showdown's ``if (!move) return false``, implemented as ``LastUsedMove::None => true`` in
``move_has_no_effect`` — failed the Encore outright. Nothing downstream could fire.

So the contract under test is a PAIR, and the pins below are deliberately written to
distinguish its two halves:

* **the world knew and said so** -> Encore applies, locks, and redirects;
* **the world genuinely had no last move** -> Encore fails, and that is CORRECT.

Collapsing those two is the failure this change must not introduce, which is why
``test_fresh_switch_in_still_fails_encore`` is a control that passes both before and after.
"""

from __future__ import annotations

import unittest

import poke_engine as pe

from pokezero.showdown import _ReplayParser


def _dummy() -> pe.Pokemon:
    return pe.Pokemon(id="pikachu", level=1, hp=0)


def _state(
    *,
    last_used_move: str | None,
    tgt_moves=("seismictoss", "rest"),
    encore_user_speed: int = 400,
    tgt_pp: int = 16,
) -> pe.State:
    """SideOne = Encore user (fast by default); SideTwo = slower target with `tgt_moves`.

    ``tgt_pp`` exists because the engine only EMITS a ``DecrementPP`` instruction once a
    slot is under 10 PP — above that the decrement is implicit and unobservable in the
    instruction list. The PP pin therefore has to run the target low deliberately; that is
    a property of the instrumentation, not of the PP rule being pinned.
    """
    enc = pe.Pokemon(
        id="jolteon", level=100, hp=300, maxhp=300, speed=encore_user_speed,
        moves=[pe.Move(id="encore", pp=16), pe.Move(id="thunderbolt", pp=16)],
    )
    tgt = pe.Pokemon(
        id="snorlax", level=100, hp=400, maxhp=400, speed=100,
        moves=[pe.Move(id=m, pp=tgt_pp) for m in tgt_moves],
    )
    kw = {"last_used_move": last_used_move} if last_used_move else {}
    return pe.State(
        side_one=pe.Side(active_index="0", pokemon=[enc] + [_dummy()] * 5),
        side_two=pe.Side(active_index="0", pokemon=[tgt] + [_dummy()] * 5, **kw),
    )


def _instructions(state: pe.State, p1: str, p2: str) -> list[str]:
    branches = pe.generate_instructions(state, p1, p2)
    return [str(i) for i in branches[0].instruction_list]


class EngineConsequencesTest(unittest.TestCase):
    """What seeding `last_used_move` actually buys — and what it must not break."""

    def test_seeded_last_move_lets_encore_apply_and_redirect_the_same_turn(self) -> None:
        # The whole point. SideTwo CHOSE `rest`; Encore lands first; SideTwo must execute
        # `seismictoss` (fixed 100) instead. This is the behaviour the 11 census rows wanted.
        out = _instructions(_state(last_used_move="move:0"), "encore", "rest")
        self.assertIn("ApplyVolatileStatus SideTwo: ENCORE", out)
        self.assertTrue(
            any("Damage SideOne: 100" in i for i in out),
            f"expected the redirected Seismic Toss to land, got {out}",
        )

    def test_world_without_a_last_move_still_fails_encore(self) -> None:
        # CONTROL — passes before and after the change, and must keep passing. Showdown's
        # `if (!move) return false` is real gen3 behaviour, not a gap. The fix must make the
        # world stop LOSING the last move; it must never invent one.
        out = _instructions(_state(last_used_move=None), "encore", "rest")
        self.assertFalse(
            any("ENCORE" in i for i in out),
            f"Encore must fail against a target with no last move, got {out}",
        )

    def test_fresh_switch_in_still_fails_encore(self) -> None:
        # The other half of the same control, and the reason the parser records a `switch`
        # SENTINEL rather than dropping to "unknown": `Pokemon.clearVolatile()` nulls
        # lastMove on switch-out, so a fresh switch-in genuinely has none. `switch` is a
        # POSITIVE fact that Encore fails against; `None` is ignorance. Both fail Encore
        # here, but only one of them is knowledge, and the world should carry which.
        out = _instructions(_state(last_used_move="switch:0"), "encore", "rest")
        self.assertFalse(
            any("ENCORE" in i for i in out),
            f"Encore must fail against a fresh switch-in, got {out}",
        )

    def test_pp_is_charged_to_the_encored_move_not_the_chosen_one(self) -> None:
        # Sim-transcribed (probed on the vendored sim: seismictoss 32 -> 30 across two turns
        # while `rest` stayed at 16, untouched). Pinned because a PP mis-charge is exactly
        # the kind of secondary divergence that surfaces as a phantom PP row two cycles later
        # with no obvious link back to Encore.
        out = _instructions(_state(last_used_move="move:0", tgt_pp=5), "encore", "rest")
        decrements = [i for i in out if "DecrementPP SideTwo" in i]
        self.assertTrue(decrements, f"expected the redirected move to be charged, got {out}")
        self.assertTrue(
            all("M0" in i for i in decrements),
            f"PP must come off the ENCORED slot (M0), not the chosen one (M1): {decrements}",
        )

    def test_encore_landing_second_does_not_redirect_that_turn(self) -> None:
        # MANDATORY control: the redirect is speed-ordered. With the Encore user SLOWER, the
        # target has already acted, so its own chosen move stands and nothing is redirected.
        # A fix that redirected unconditionally would pass every test above and be wrong.
        out = _instructions(
            _state(last_used_move="move:0", encore_user_speed=1), "encore", "rest"
        )
        self.assertFalse(
            any("Damage SideOne: 100" in i for i in out),
            f"a second-moving Encore must not redirect the same turn, got {out}",
        )

    def test_fake_out_is_the_only_other_consumer_and_it_is_pool_unreachable(self) -> None:
        # Consumer survey, pinned. Seeding `last_used_move` newly reaches exactly one
        # non-Encore behaviour in the gen3 engine: `choice_effects.rs` strips ALL of Fake
        # Out's effects when the user has already moved. Spite / Grudge / Mirror Move do not
        # read `last_used_move` in gen3 at all.
        #
        # Fake Out is NOT in the gen3 randbats pool, so this cannot fire in randbats — but it
        # IS reachable in gen3customgame, which the fixture harness uses, and it changes
        # behaviour silently. Pinned so the change is recorded rather than discovered.
        fresh = pe.Pokemon(
            id="hitmonchan", level=100, hp=300, maxhp=300, speed=400,
            moves=[pe.Move(id="fakeout", pp=16)],
        )
        victim = pe.Pokemon(
            id="snorlax", level=100, hp=400, maxhp=400, speed=100,
            moves=[pe.Move(id="splash", pp=16)],
        )

        def flinches(last: str | None) -> bool:
            kw = {"last_used_move": last} if last else {}
            st = pe.State(
                side_one=pe.Side(active_index="0", pokemon=[fresh] + [_dummy()] * 5, **kw),
                side_two=pe.Side(active_index="0", pokemon=[victim] + [_dummy()] * 5),
            )
            return any(
                "FLINCH" in str(i)
                for b in pe.generate_instructions(st, "fakeout", "splash")
                for i in b.instruction_list
            )

        self.assertTrue(flinches(None), "a just-switched-in Fake Out user should still flinch")
        self.assertFalse(
            flinches("move:0"),
            "Fake Out must lose its effects once the user has already used a move",
        )


class ParserTruthTableTest(unittest.TestCase):
    """The parser half, transcribed from the same table the ENGINE already obeys.

    ``third_party/poke-engine-gen3-lastmove-semantics.patch`` moved the engine's record point
    to match ``Pokemon.moveUsed()``. These pins make the parser agree with it, because a
    world that disagrees with the engine about what "last move" means is worse than one that
    omits it — it would lock Encore onto the wrong move rather than not at all.
    """

    def _parse(self, lines: list[str]) -> _ReplayParser:
        parser = _ReplayParser("battle-gen3randombattle-1")
        parser.feed([
            "|player|p1|Us|",
            "|player|p2|Them|",
            "|switch|p1a: Skarmory|Skarmory, M|215/215",
            "|switch|p2a: Illumise|Illumise, F|269/269",
            *lines,
        ])
        return parser

    def test_an_executed_move_is_recorded(self) -> None:
        p = self._parse(["|move|p1a: Skarmory|Drill Peck|p2a: Illumise"])
        self.assertEqual(p.last_used_move["p1"], "drillpeck")

    def test_a_move_that_missed_or_failed_still_counts_as_used(self) -> None:
        # `moveUsed` precedes `useMove`, so the outcome is irrelevant — Showdown emits the
        # `|move|` line either way. Recording on `|move|` is therefore exactly right, and
        # this pin is what stops someone "fixing" it to require a successful outcome.
        miss = self._parse([
            "|move|p1a: Skarmory|Drill Peck|p2a: Illumise|[miss]",
            "|-miss|p1a: Skarmory|p2a: Illumise",
        ])
        self.assertEqual(miss.last_used_move["p1"], "drillpeck")
        fail = self._parse([
            "|move|p1a: Skarmory|Protect||[still]",
            "|-fail|p1a: Skarmory",
        ])
        self.assertEqual(fail.last_used_move["p1"], "protect")

    def test_an_immobilized_turn_records_nothing(self) -> None:
        # Every immobilizer returns false from onBeforeMove, so no `|move|` line is emitted
        # at all — Showdown emits `|cant|`. The parser matches by construction, and this pin
        # holds that: a paralysed turn must not overwrite the real last move.
        p = self._parse([
            "|move|p1a: Skarmory|Drill Peck|p2a: Illumise",
            "|cant|p1a: Skarmory|par",
        ])
        self.assertEqual(p.last_used_move["p1"], "drillpeck")

    def test_sleep_talk_records_the_caller_not_the_callee(self) -> None:
        # The inversion the engine-side patch explicitly called out. Sleep Talk is
        # `sleepUsable` and reaches moveUsed; the move it CALLS goes through `useMove`, which
        # never touches lastMove. Publicly, the callee's line carries `[from]`.
        # If this were backwards, Encore would lock onto the called move.
        p = self._parse([
            "|move|p1a: Skarmory|Sleep Talk|p1a: Skarmory",
            "|move|p1a: Skarmory|Drill Peck|p2a: Illumise|[from]Sleep Talk",
        ])
        self.assertEqual(p.last_used_move["p1"], "sleeptalk")

    def test_a_lockedmove_continuation_does_NOT_advance_the_latch(self) -> None:
        """The ONE place this latch knowingly disagrees with Showdown's ``lastMove``.

        ``runMove`` (``sim/battle-actions.ts``) takes the ``getLockedMove()`` branch,
        sets ``sourceEffect = lockedmove``, and then STILL calls
        ``pokemon.moveUsed(move, targetLoc)`` -- so a locked continuation DOES advance
        ``lastMove``. It also emits ``|[from] lockedmove`` (``if (sourceEffect) attrs +=
        ...``). This parser rejects EVERY ``[from]``, so on that line the latch keeps
        the CALLER and Showdown has already moved on to the callee.

        The divergence is conservative in the only direction that matters here -- the
        latch lags, it does not invent -- and it is pinned rather than fixed, because
        the two consumers want different rules: this latch feeds
        ``LastUsedMove`` seeding, while ``determinization._called_move_line``
        deliberately TOLERATES ``lockedmove`` (``if "lockedmove" in normalized:
        continue``) so the Encore event scan reports the faithful value.

        Sleep Talk calling a locking move is the shape that separates them, and it is
        the sharpest case available: the callee's line is BOTH ``[from]``-tagged and a
        real ``moveUsed``. See ``PoolCannotReachTheLockedMoveDivergenceTests`` in
        ``tests/test_engine_world_encore_last_used.py`` for why no gen3 randbats set can
        produce it, and why precedence would cover it even if one could.
        """

        p = self._parse([
            "|move|p1a: Skarmory|Sleep Talk|p1a: Skarmory",
            "|move|p1a: Skarmory|Thrash|p2a: Illumise|[from]lockedmove",
        ])
        self.assertEqual(p.last_used_move["p1"], "sleeptalk")

        # The OTHER half, and the reason this is a divergence rather than a quirk: the
        # Encore event scan reads the very same line and returns Thrash. Asserting only
        # the parser side would pin agreement that does not exist.
        from pokezero.determinization import _move_from_public_event_line

        self.assertEqual(
            _move_from_public_event_line(
                "|move|p1a: Skarmory|Thrash|p2a: Illumise|[from]lockedmove",
                opponent_slot="p1",
                self_slot="p2",
                species="Skarmory",
            ),
            "Thrash",
        )
        # Control: the two rules AGREE on an ordinary called move, so the assertion
        # above isolates `lockedmove` and not `[from]` in general.
        self.assertIsNone(
            _move_from_public_event_line(
                "|move|p1a: Skarmory|Drill Peck|p2a: Illumise|[from]Sleep Talk",
                opponent_slot="p1",
                self_slot="p2",
                species="Skarmory",
            )
        )

    def test_switching_out_clears_to_the_switch_sentinel(self) -> None:
        p = self._parse([
            "|move|p1a: Skarmory|Drill Peck|p2a: Illumise",
            "|switch|p1a: Blissey|Blissey, F|300/300",
        ])
        self.assertEqual(p.last_used_move["p1"], "switch")

    def test_the_two_sides_are_tracked_independently(self) -> None:
        p = self._parse([
            "|move|p1a: Skarmory|Drill Peck|p2a: Illumise",
            "|move|p2a: Illumise|Encore|p1a: Skarmory",
        ])
        self.assertEqual(p.last_used_move["p1"], "drillpeck")
        self.assertEqual(p.last_used_move["p2"], "encore")


if __name__ == "__main__":
    unittest.main()
