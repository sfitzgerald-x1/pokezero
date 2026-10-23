"""Authored public protocol only; no historical state replay or strength claim."""
import unittest

from pokezero.belief import PublicBattleBeliefEngine
from pokezero.showdown import parse_showdown_replay


def engine(lines):
    return PublicBattleBeliefEngine.from_events(parse_showdown_replay(
        ["|start", *lines], battle_id="authored-running-ability-effect").public_events)


def mon(state, side, species):
    return next(m for m in state.snapshot().side(side) if m.species == species)


def trace_lines(side="p1", *, copied="Water Absorb", target="Lapras"):
    other = "p2" if side == "p1" else "p1"
    return [f"|switch|{side}a: Porygon2|Porygon2, L80|100/100",
            f"|switch|{other}a: {target}|{target}, L80|100/100",
            f"|-ability|{side}a: Porygon2|{copied}|Trace|[from] ability: Trace|[of] {other}a: {target}",
            "|turn|1"]


class RunningAbilityEffectsTests(unittest.TestCase):
    def test_traced_water_absorb_proc_never_becomes_original_set_evidence(self):
        for side, other in (("p1", "p2"), ("p2", "p1")):
            with self.subTest(side=side):
                state = engine([*trace_lines(side),
                    f"|-heal|{side}a: Porygon2|100/100|[from] ability: Water Absorb|[of] {other}a: Lapras"])
                tracer = mon(state, side, "Porygon2")
                self.assertIsNone(tracer.revealed_ability)
                self.assertEqual(state._running_ability[tracer.key], "Water Absorb")
                self.assertEqual(mon(state, other, "Lapras").revealed_ability, "Water Absorb")

    def test_traced_immunity_proc_does_not_create_an_original_ability(self):
        state = engine([*trace_lines(copied="Volt Absorb", target="Lanturn"),
                        "|-immune|p1a: Porygon2|[from] ability: Volt Absorb"])
        self.assertIsNone(mon(state, "p1", "Porygon2").revealed_ability)

    def test_traced_ability_activation_leaves_existing_original_reveal_unchanged(self):
        lines = trace_lines(copied="Flash Fire", target="Houndoom")
        lines.insert(2, "|-ability|p1a: Porygon2|Trace|[silent]")
        state = engine([*lines, "|-activate|p1a: Porygon2|ability: Flash Fire"])
        tracer = mon(state, "p1", "Porygon2")
        self.assertEqual(tracer.revealed_ability, "Trace")
        self.assertFalse(any(e.kind == "conflicting-ability-evidence" for e in tracer.evidence))
        self.assertEqual(state._running_ability[tracer.key], "Flash Fire")

    def test_reentry_uses_new_copy_without_suppressing_real_holder(self):
        state = engine([*trace_lines(),
            "|-heal|p1a: Porygon2|100/100|[from] ability: Water Absorb|[of] p2a: Lapras",
            "|switch|p1a: Snorlax|Snorlax, L80|100/100",
            "|-heal|p2a: Lapras|100/100|[from] ability: Water Absorb|[of] p1a: Snorlax",
            "|switch|p2a: Lanturn|Lanturn, L80|100/100",
            "|switch|p1a: Porygon2|Porygon2, L80|100/100",
            "|-ability|p1a: Porygon2|Volt Absorb|Trace|[from] ability: Trace|[of] p2a: Lanturn",
            "|-heal|p1a: Porygon2|100/100|[from] ability: Volt Absorb|[of] p2a: Lanturn"])
        tracer = mon(state, "p1", "Porygon2")
        self.assertIsNone(tracer.revealed_ability)
        self.assertEqual(state._running_ability[tracer.key], "Volt Absorb")
        self.assertEqual(mon(state, "p2", "Lapras").revealed_ability, "Water Absorb")
        self.assertEqual(mon(state, "p2", "Lanturn").revealed_ability, "Volt Absorb")

    def test_known_transformed_running_ability_is_also_not_original_set_evidence(self):
        state = engine([
            "|switch|p1a: Ditto|Ditto, L80|100/100",
            "|switch|p2a: Lapras|Lapras, L80|100/100",
            "|-ability|p2a: Lapras|Water Absorb|[silent]",
            "|-transform|p1a: Ditto|p2a: Lapras",
            "|-heal|p1a: Ditto|100/100|[from] ability: Water Absorb|[of] p2a: Lapras"])
        ditto = mon(state, "p1", "Ditto")
        self.assertTrue(ditto.transformed)
        self.assertEqual(state._running_ability[ditto.key], "Water Absorb")
        self.assertIsNone(ditto.revealed_ability)

    def test_ordinary_uncopied_ability_procs_still_identify_original_holder(self):
        state = engine([
            "|switch|p1a: Suicune|Suicune, L80|100/100",
            "|switch|p2a: Quagsire|Quagsire, L80|100/100",
            "|-heal|p2a: Quagsire|100/100|[from] ability: Water Absorb|[of] p1a: Suicune"])
        self.assertEqual(mon(state, "p2", "Quagsire").revealed_ability, "Water Absorb")
        self.assertIsNone(mon(state, "p1", "Suicune").revealed_ability)


if __name__ == "__main__":
    unittest.main()
