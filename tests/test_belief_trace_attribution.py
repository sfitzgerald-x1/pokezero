"""Trace attributes the copied ability to the TRACED mon, not the tracer.

``sim/pokemon.ts setAbility`` emits

    |-ability|<tracer>|<COPIED>|<tracer's own ability>|[from] ability: Trace|[of] <traced>

so the line's SUBJECT is the tracer while the ability named belongs to the TRACED mon.
Attributing it to the subject is wrong twice over: it corrupts the tracer's candidate set with
an ability it cannot have, and it discards a certain reveal that narrows the traced mon.

The corruption is sticky, which is what makes it worth a regression test: Trace re-fires on
every switch-in and the conflicting-ability-evidence guard keeps the FIRST claim, so one early
copy would persist for the whole battle.

Reachable in the gen3 randbats pool via Gardevoir and Porygon2.
"""
import unittest

from _showdown_root import requires_showdown, showdown_root_str

from pokezero.belief import PublicBattleBeliefEngine
from pokezero.showdown import parse_showdown_replay


def _engine(lines):
    return PublicBattleBeliefEngine.from_events(
        parse_showdown_replay(lines, battle_id="battle-gen3randombattle-trace").public_events
    )


def _ability(mon) -> str:
    """Normalized ability id — `revealed_ability` stores the DISPLAY name ("Keen Eye")."""
    return (mon.revealed_ability or "").lower().replace(" ", "")


def _belief(engine, slot, species):
    for mon in engine.snapshot().side(slot):
        if mon.species.lower().startswith(species.lower()):
            return mon
    raise AssertionError(f"no {species} on {slot}")


OPENING = [
    "|start",
    "|switch|p1a: Gardevoir|Gardevoir, L79, F|100/100",
    "|switch|p2a: Claydol|Claydol, L81|100/100",
]
TRACE_LINE = (
    "|-ability|p1a: Gardevoir|Levitate|Trace|[from] ability: Trace|[of] p2a: Claydol"
)


class TraceAttributionTest(unittest.TestCase):
    def test_copied_ability_lands_on_the_traced_mon(self) -> None:
        engine = _engine([*OPENING, TRACE_LINE, "|turn|1"])
        self.assertEqual(
            _ability(_belief(engine, "p2", "Claydol")), "levitate"
        )

    def test_tracer_is_not_credited_with_the_copied_ability(self) -> None:
        """The live bug: Gardevoir holding `levitate` and gaining Spikes immunity."""
        engine = _engine([*OPENING, TRACE_LINE, "|turn|1"])
        self.assertNotEqual(
            _ability(_belief(engine, "p1", "Gardevoir")), "levitate"
        )

    def test_a_later_trace_is_not_blocked_by_a_stale_first_copy(self) -> None:
        """Trace re-fires on switch-in; each copy must reach its own traced mon."""
        engine = _engine(
            [
                *OPENING,
                TRACE_LINE,
                "|turn|1",
                "|switch|p2a: Skarmory|Skarmory, L79, M|100/100",
                "|-ability|p1a: Gardevoir|Keen Eye|Trace|[from] ability: Trace|[of] p2a: Skarmory",
                "|turn|2",
            ]
        )
        self.assertEqual(
            _ability(_belief(engine, "p2", "Claydol")), "levitate"
        )
        self.assertEqual(
            _ability(_belief(engine, "p2", "Skarmory")), "keeneye"
        )
        self.assertNotEqual(
            _ability(_belief(engine, "p1", "Gardevoir")), "keeneye"
        )

    def test_missing_of_tag_records_nothing_rather_than_the_subject(self) -> None:
        """No attribution is available, and guessing the subject is the original bug."""
        engine = _engine(
            [*OPENING, "|-ability|p1a: Gardevoir|Levitate|Trace|[from] ability: Trace", "|turn|1"]
        )
        self.assertNotEqual(
            _ability(_belief(engine, "p1", "Gardevoir")), "levitate"
        )

    def test_ordinary_ability_reveals_still_attribute_to_the_subject(self) -> None:
        """Guard the narrowness of the change: only Trace-sourced lines flip."""
        engine = _engine(
            [
                "|start",
                "|switch|p1a: Zapdos|Zapdos, L77|100/100",
                "|switch|p2a: Claydol|Claydol, L81|100/100",
                "|-ability|p1a: Zapdos|Pressure",
                "|turn|1",
            ]
        )
        self.assertEqual(
            _ability(_belief(engine, "p1", "Zapdos")), "pressure"
        )


class TraceOnTransformedTargetTest(unittest.TestCase):
    """Trace copies what the target is CURRENTLY running, which for a transformed mon is its
    copy target's ability, not its own.

    Recording it as a certain reveal writes a fact the mon cannot have: a Ditto transformed into
    a Levitate holder yields `revealed_ability=Levitate` on Ditto, whose only pool ability is
    Limber, collapsing its candidate set to the inconsistent fallback. It widens rather than
    dropping the true variant, so it is safe -- but a wrong sticky fact is worse than no fact.
    """

    LINES = [
        "|start",
        "|switch|p1a: Gardevoir|Gardevoir, L79, F|100/100",
        "|switch|p2a: Ditto|Ditto, L83|100/100",
        "|-transform|p2a: Ditto|p1a: Gardevoir",
        "|switch|p1a: Gardevoir|Gardevoir, L79, F|100/100",
        "|-ability|p1a: Gardevoir|Levitate|Trace|[from] ability: Trace|[of] p2a: Ditto",
        "|turn|1",
    ]

    def test_a_transformed_target_receives_no_ability_reveal(self) -> None:
        engine = _engine(self.LINES)
        ditto = _belief(engine, "p2", "Ditto")
        self.assertTrue(ditto.transformed, "fixture must actually transform, or this proves nothing")
        self.assertIsNone(ditto.revealed_ability)

    def test_a_normal_target_still_receives_it(self) -> None:
        """Guard the narrowness: only the transformed case is suppressed."""
        lines = [
            "|start",
            "|switch|p1a: Gardevoir|Gardevoir, L79, F|100/100",
            "|switch|p2a: Claydol|Claydol, L81|100/100",
            "|-ability|p1a: Gardevoir|Levitate|Trace|[from] ability: Trace|[of] p2a: Claydol",
            "|turn|1",
        ]
        self.assertEqual(_ability(_belief(_engine(lines), "p2", "Claydol")), "levitate")


class TraceOfATracerTest(unittest.TestCase):
    """A tracer that has already copied something is RUNNING that, not its own ability.

    Trace carries `notrace`, so a tracer cannot copy Trace itself -- but it can copy whatever a
    tracer is currently running, and the `[of]` redirect then writes that onto the FIRST tracer as
    a certain reveal. It is a fact that mon cannot have: Gardevoir's only pool ability is Trace.

    The damage is the sticky full-pool collapse this module exists to prevent. `randbat.py` filters
    every Gardevoir variant against `revealed_ability=Pressure`, finds none, and falls back to the
    whole species pool -- and the conflicting-ability guard then keeps that wrong claim for the
    rest of the battle.

    Reachable: the gen3 randbats pool has exactly two Trace carriers, Porygon2 and Gardevoir, so
    one Tracing the other is an ordinary game. Found in independent review; the sibling guard for
    the TRANSFORM producer was already present eight lines away.
    """

    LINES = [
        "|start",
        "|switch|p1a: Gardevoir|Gardevoir, L79, F|100/100",
        "|switch|p2a: Absol|Absol, L81, M|100/100",
        # Gardevoir Traces Absol: it is now RUNNING Pressure.
        "|-ability|p1a: Gardevoir|Pressure|Trace|[from] ability: Trace|[of] p2a: Absol",
        # One revealed move, so Gardevoir is NARROWED (10 pool variants -> 4) before the second
        # Trace lands. Without this the mon sits at the full pool in both worlds and the collapse
        # assertion below cannot distinguish them -- which is exactly how the first version of
        # this test managed to measure nothing.
        "|move|p1a: Gardevoir|Calm Mind|p1a: Gardevoir",
        "|turn|1",
        "|switch|p2a: Porygon2|Porygon2, L80|100/100",
        # Porygon2 Traces Gardevoir and copies what Gardevoir is RUNNING, not what it has.
        "|-ability|p2a: Porygon2|Pressure|Trace|[from] ability: Trace|[of] p1a: Gardevoir",
        "|turn|2",
    ]

    def test_the_second_trace_does_not_write_a_false_ability_onto_the_first_tracer(self) -> None:
        gardevoir = _belief(_engine(self.LINES), "p1", "Gardevoir")
        self.assertNotEqual(
            _ability(gardevoir),
            "pressure",
            "the second Trace wrote Absol's Pressure onto Gardevoir, whose only pool ability is "
            "Trace -- a fact it cannot have, and a sticky one",
        )

    # Decorated per-METHOD, not on the class. The sibling test below needs no checkout -- it is
    # pure fixture -- and CI has none (`.github/workflows/engine-fidelity-gates.yml`). A
    # class-level decorator skipped BOTH there, taking this belief fix from one executing
    # regression test in CI to zero: worse than before the decorator was added.
    @requires_showdown("the collapse is measured against the real randbats set source")
    def test_the_first_tracer_is_not_collapsed_to_the_full_pool(self) -> None:
        """The consequence that matters: a false ability filters out every real variant.

        Uses the REAL randbats source, because the collapse happens in `randbat.py:262` --
        `if revealed_ability and _normalize_id(self.ability) != _normalize_id(revealed_ability)`.
        Without a set source `candidate_variants` is `()` unconditionally and any comparison
        between the two worlds is `0 <= 0`; the first version of this test did exactly that and
        measured nothing, which independent review caught.
        """
        from pokezero.randbat import Gen3RandbatSource

        source = Gen3RandbatSource.from_showdown_root(showdown_root_str())

        def gardevoir_after(lines):
            engine = PublicBattleBeliefEngine.from_events(
                parse_showdown_replay(
                    lines, battle_id="battle-gen3randombattle-trace"
                ).public_events,
                format_id="gen3randombattle",
                set_source=source,
            )
            return _belief(engine, "p1", "Gardevoir")

        without_second = gardevoir_after(self.LINES[:6])
        # Measured: 4 of Gardevoir's 10 pool variants survive one Calm Mind reveal.
        with_second = gardevoir_after(self.LINES)

        # Precondition: the mon must be NARROWED, not merely non-empty. `> 0` was too weak -- if a
        # randbats change ever put Calm Mind on all ten Gardevoir variants, `without_second` would
        # be 10, the bug also yields 10, and `10 == 10` would pass while measuring nothing. That is
        # the round-4 finding re-armed, so the bound is the full pool.
        full_pool = len(
            source.summarize(
                format_id="gen3randombattle", species="Gardevoir", revealed_moves=()
            ).candidate_variants
        )
        self.assertGreater(full_pool, 0, "the set source resolved no Gardevoir variants at all")
        self.assertLess(
            len(without_second.candidate_variants),
            full_pool,
            "the Calm Mind reveal did not narrow Gardevoir below its full pool, so this test "
            "cannot distinguish the collapse it exists to catch",
        )
        self.assertEqual(
            len(with_second.candidate_variants),
            len(without_second.candidate_variants),
            "the second Trace changed the first tracer's candidate set -- writing a false "
            "revealed_ability filters every real variant out and falls back to the full pool",
        )


if __name__ == "__main__":  # pragma: no cover
    # At the END. It sat above TraceOnTransformedTargetTest and TraceOfATracerTest, so direct
    # execution ran 5 of 9 tests and never even DEFINED the two classes carrying this change's
    # kill-confirmation. Round 3 fixed exactly this defect in test_pressure_pp_charge.py, and I
    # then added new tests below the same defect in this file.
    unittest.main()
