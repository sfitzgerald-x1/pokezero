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


if __name__ == "__main__":
    unittest.main()


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
