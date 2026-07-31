"""Two world-seeding gaps, both of the "parser knew, world didn't listen" class.

W1 — **traced ability**. `|-ability|<mon>|<Ability>|[from] ability: Trace` publicly replaces
the holder's ability. The world kept rebuilding from the sampled set, handing the engine
`TRACE` and playing the mon without the copied ability at all.

The fix is tracked in the PARSER, not read from `belief.revealed_ability`, and the difference
is the whole lesson. The belief field is right for an ability a mon merely revealed — that is
a permanent fact. A traced ability is transient: Trace re-fires on every switch-in and the
copy dies on switch-out, so the belief holds the last ability that mon EVER traced. The first
version of this change read it and stamped a historical trace, handing a Gardevoir `levitate`
from an earlier switch-in and silently granting it Spikes immunity — a three-row fix that
arrived with a two-row regression. `test_a_stale_trace_never_leaks_into_a_later_switch_in`
pins that specific failure.

W2 — **toxic-stage staleness**. `_update_toxic_stage` reset the ramp on `-curestatus` and
`-cureteam` only. `Pokemon.setStatus` replaces `statusState` wholesale
(`sim/pokemon.ts:1733`) and emits NO cure line for the status it displaced, so Rest putting a
badly-poisoned mon to sleep left the ramp standing at a stage that no longer existed.

Both halves are pinned separately, because they fail differently: a parser pin failing means
the world is being told something false; a world pin failing means it was told the truth and
ignored it.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

from pokezero.local_showdown import (
    _apply_traced_ability_materialization_state,
    _materialization_toxic_stage,
    _seed_scenario_parser_state,
)
from pokezero.showdown import ShowdownPokemon, _ReplayParser, _update_toxic_stage


class ToxicStageParserTest(unittest.TestCase):
    """W2, parser half — which status lines end the ramp."""

    @staticmethod
    def _percentage_reentry_parser(event: str, *, include_second_tick: bool = False) -> _ReplayParser:
        """Return a public `/100` Toxic re-entry through an ordinary or forced switch.

        The rounded stream cannot reveal the exact HP unit, but the switch/drag
        line proves Gen 3 reset the Toxic counter. The first residual after that
        public reset therefore proves stage one; the second proves stage two.
        """

        parser = _ReplayParser(
            f"toxic-percentage-{event}",
            complete_prefix=True,
            hp_visibility={"p1": "percentage", "p2": "percentage"},
        )
        parser.feed(
            [
                "|switch|p1a: Tauros|Tauros, L80, M|100/100",
                "|switch|p2a: Milotic|Milotic, L80, F|100/100",
                "|turn|1",
                "|-status|p1a: Tauros|tox",
                "|-damage|p1a: Tauros|95/100 tox|[from] psn",
                "|upkeep",
                "|turn|2",
                f"|{event}|p1a: Zapdos|Zapdos, L78, M|100/100",
                "|upkeep",
                "|turn|3",
                f"|{event}|p1a: Tauros|Tauros, L80, M|90/100 tox",
                "|-damage|p1a: Tauros|85/100 tox|[from] psn",
            ]
        )
        if include_second_tick:
            parser.feed(
                [
                    "|upkeep",
                    "|turn|4",
                    "|-damage|p1a: Tauros|75/100 tox|[from] psn",
                ]
            )
        return parser

    def test_a_fresh_tox_starts_the_ramp_at_one(self) -> None:
        stage = {"p1": 0, "p2": 0}
        _update_toxic_stage(["", "-status", "p1a: Zapdos", "tox"], stage)
        self.assertEqual(stage["p1"], 1)

    def test_percentage_reentry_first_tick_reseeds_and_materializes_after_switch_or_drag(self) -> None:
        # The parser stage is the residual-side feature convention. The world
        # consumes one less at a request boundary, so turn 4's stage 2 becomes
        # the engine's ToxicCount 1 after the first post-entry residual.
        for event in ("switch", "drag"):
            with self.subTest(event=event):
                parser = self._percentage_reentry_parser(event)
                self.assertEqual(parser.toxic_stage["p1"], 1)
                self.assertTrue(parser.toxic_stage_known["p1"])
                parser.feed(["|upkeep", "|turn|4"])
                self.assertEqual(parser.toxic_stage["p1"], 2)
                self.assertEqual(_materialization_toxic_stage(parser.snapshot(), "p1"), 1)

    def test_percentage_reentry_second_tick_preserves_the_public_ramp_after_switch_or_drag(self) -> None:
        for event in ("switch", "drag"):
            with self.subTest(event=event):
                parser = self._percentage_reentry_parser(event, include_second_tick=True)
                self.assertEqual(parser.toxic_stage["p1"], 2)
                self.assertTrue(parser.toxic_stage_known["p1"])
                parser.feed(["|upkeep", "|turn|5"])
                self.assertEqual(parser.toxic_stage["p1"], 3)
                self.assertEqual(_materialization_toxic_stage(parser.snapshot(), "p1"), 2)

    def test_a_replacing_status_ends_the_ramp(self) -> None:
        # THE FIX. Rest is the reachable case: `|-status|<mon>|slp|[from] move: Rest` on an
        # already-toxed mon. Showdown emits no `-curestatus` for the tox it displaced, so
        # before this the ramp survived a status it no longer belonged to and a LATER re-tox
        # was priced from it (observed: a stage-5 tick of -75 where Showdown ticked -15).
        for replacing in ("slp", "brn", "par", "frz", "psn"):
            stage = {"p1": 6, "p2": 0}
            _update_toxic_stage(["", "-status", "p1a: Zapdos", replacing], stage)
            self.assertEqual(stage["p1"], 0, f"{replacing} must end the toxic ramp")

    def test_an_explicit_cure_still_ends_the_ramp(self) -> None:
        # Pre-existing behaviour, pinned so the new arm cannot be written in a way that
        # swallows it.
        for event in ("-curestatus", "-cureteam"):
            stage = {"p1": 4, "p2": 0}
            _update_toxic_stage(["", event, "p1a: Zapdos", "tox"], stage)
            self.assertEqual(stage["p1"], 0)

    def test_natural_cure_ends_the_active_ramp(self) -> None:
        # Natural Cure's switch-out cure is public, but it does not reveal any
        # private statusState counter. The cure line is enough to retire it.
        stage = {"p1": 4, "p2": 0}
        known = {"p1": True, "p2": True}
        _update_toxic_stage(
            ["", "-curestatus", "p1a: Zapdos", "tox", "[silent]", "[from] ability: Natural Cure"],
            stage,
            known,
        )
        self.assertEqual(stage["p1"], 0)
        self.assertTrue(known["p1"])

    def test_active_status_lifecycle_retires_post_upkeep_zero_proof(self) -> None:
        # The proof is for one specific active Toxic stint, not an assertion
        # about the slot forever. All of the regular active-status reset paths
        # must clear it alongside the numeric ramp.
        for event in (
            ["", "-status", "p1a: Zapdos", "tox"],
            ["", "-status", "p1a: Zapdos", "slp"],
            ["", "-curestatus", "p1a: Zapdos", "tox"],
            ["", "-cureteam", "p1a: Zapdos", "tox"],
            ["", "faint", "p1a: Zapdos"],
        ):
            with self.subTest(event=event[1]):
                stage = {"p1": 0, "p2": 0}
                known = {"p1": True, "p2": True}
                zero_after_upkeep = {"p1": True, "p2": False}
                _update_toxic_stage(event, stage, known, zero_after_upkeep)
                self.assertFalse(zero_after_upkeep["p1"])

    def test_failed_reapplication_does_not_rewrite_the_live_counter(self) -> None:
        # Re-Toxic into an already-toxic target is a public failure, not a new
        # statusState. A later cure plus fresh -status is the only re-seed.
        stage = {"p1": 4, "p2": 0}
        known = {"p1": True, "p2": True}
        _update_toxic_stage(["", "-fail", "p1a: Zapdos", "move: Toxic"], stage, known)
        self.assertEqual(stage["p1"], 4)
        self.assertTrue(known["p1"])
        _update_toxic_stage(["", "-curestatus", "p1a: Zapdos", "tox"], stage, known)
        _update_toxic_stage(["", "-status", "p1a: Zapdos", "tox"], stage, known)
        self.assertEqual(stage["p1"], 1)

    def test_benched_cure_does_not_reset_the_active_toxic_counter(self) -> None:
        # Heal Bell emits a position-less ``p1: Name`` cure for a benched ally.
        # Its statusState is not the active mon's Toxic statusState.
        stage = {"p1": 5, "p2": 0}
        _update_toxic_stage(["", "-curestatus", "p1: Blissey", "par", "[silent]"], stage)
        self.assertEqual(stage["p1"], 5)

    def test_faint_retires_the_counter_before_replacement(self) -> None:
        stage = {"p1": 5, "p2": 0}
        known = {"p1": True, "p2": True}
        _update_toxic_stage(["", "faint", "p1a: Zapdos"], stage, known)
        self.assertEqual(stage["p1"], 0)
        self.assertTrue(known["p1"])

    def test_the_other_side_is_untouched(self) -> None:
        stage = {"p1": 5, "p2": 3}
        _update_toxic_stage(["", "-status", "p1a: Zapdos", "slp"], stage)
        self.assertEqual(stage["p2"], 3)

    def test_rest_then_retox_prices_from_stage_one(self) -> None:
        """End-to-end through the real parser: the exact 1500243/79 sequence."""
        parser = _ReplayParser("battle-gen3randombattle-1")
        parser.feed([
            "|player|p1|Us|",
            "|player|p2|Them|",
            "|switch|p1a: Whiscash|Whiscash, M|318/318",
            "|switch|p2a: Zapdos|Zapdos|255/255",
            "|move|p1a: Whiscash|Toxic|p2a: Zapdos",
            "|-status|p2a: Zapdos|tox",
            "|turn|2",
            "|turn|3",
            "|turn|4",
            # Rest displaces the tox with sleep, and emits no cure line for it.
            "|move|p2a: Zapdos|Rest|p2a: Zapdos",
            "|-status|p2a: Zapdos|slp|[from] move: Rest",
            "|turn|5",
            # Woken, then poisoned again: this must be a FRESH stage 1.
            "|-curestatus|p2a: Zapdos|slp|[msg]",
            "|move|p1a: Whiscash|Toxic|p2a: Zapdos",
            "|-status|p2a: Zapdos|tox",
        ])
        self.assertEqual(parser.toxic_stage["p2"], 1)


class ToxicStageWorldTest(unittest.TestCase):
    """W2, world half — translate public chronology at the exact boundary."""

    def test_turn_and_post_upkeep_boundaries_use_the_simulators_current_stage(self) -> None:
        parser = _ReplayParser(
            "toxic-boundary",
            complete_prefix=True,
            hp_visibility={"p1": "exact", "p2": "exact"},
        )
        parser.feed(
            [
                "|switch|p1a: Diglett|Diglett, L60, M|100/100",
                "|switch|p2a: Magikarp|Magikarp, L100, M|181/181",
                "|turn|1",
                "|-status|p1a: Diglett|tox",
                "|-damage|p1a: Diglett|94/100 tox|[from] psn",
                "|upkeep",
            ]
        )

        post_upkeep = parser.snapshot()
        self.assertTrue(post_upkeep.post_upkeep_window)
        self.assertEqual(post_upkeep.toxic_stage["p1"], 1)
        self.assertEqual(_materialization_toxic_stage(post_upkeep, "p1"), 1)

        parser.feed(["|turn|2"])
        ordinary = parser.snapshot()
        self.assertFalse(ordinary.post_upkeep_window)
        self.assertEqual(ordinary.toxic_stage["p1"], 2)
        self.assertEqual(_materialization_toxic_stage(ordinary, "p1"), 1)
        self.assertEqual(_materialization_toxic_stage(ordinary, "p2"), 0)

    def test_post_upkeep_poisoned_replacement_materializes_zero_for_both_sides(self) -> None:
        """A forced post-residual replacement has not yet paid its first Toxic tick."""
        for replaced_side in ("p1", "p2"):
            with self.subTest(replaced_side=replaced_side):
                parser = _ReplayParser(
                    f"toxic-post-upkeep-{replaced_side}",
                    complete_prefix=True,
                    hp_visibility={"p1": "exact", "p2": "exact"},
                )
                other_side = "p2" if replaced_side == "p1" else "p1"
                parser.feed(
                    [
                        "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                        "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                        "|turn|1",
                        f"|faint|{replaced_side}a: {'LeadOne' if replaced_side == 'p1' else 'LeadTwo'}",
                        "|upkeep",
                        (
                            f"|switch|{replaced_side}a: Replacement|Replacement, L80, M|"
                            "90/100 tox"
                        ),
                        "|turn|2",
                        f'|request|{{"side":{{"id":"{replaced_side}"}}}}',
                    ]
                )

                replay = parser.snapshot()
                self.assertFalse(replay.post_upkeep_window)
                self.assertFalse(replay.toxic_faint_replacement_pending[replaced_side])
                self.assertEqual(replay.toxic_stage[replaced_side], 0)
                self.assertTrue(replay.toxic_stage_known[replaced_side])
                self.assertTrue(replay.toxic_stage_zero_after_upkeep[replaced_side])
                self.assertEqual(_materialization_toxic_stage(replay, replaced_side), 0)
                self.assertEqual(_materialization_toxic_stage(replay, other_side), 0)

                resumed = _ReplayParser.from_snapshot(replay).snapshot()
                self.assertTrue(resumed.toxic_stage_zero_after_upkeep[replaced_side])
                self.assertEqual(_materialization_toxic_stage(resumed, replaced_side), 0)

                # A legacy snapshot lacks the new proof. Its numeric zero must
                # remain fail-closed rather than reinterpret an old checkpoint.
                legacy = replace(replay, toxic_stage_zero_after_upkeep={})
                legacy_resumed = _ReplayParser.from_snapshot(legacy).snapshot()
                self.assertFalse(legacy_resumed.toxic_stage_zero_after_upkeep[replaced_side])
                self.assertIsNone(_materialization_toxic_stage(legacy_resumed, replaced_side))

    def test_real_damage_faint_replacement_chronology_proves_zero_for_both_sides(self) -> None:
        for replaced_side in ("p1", "p2"):
            with self.subTest(replaced_side=replaced_side):
                lead = "LeadOne" if replaced_side == "p1" else "LeadTwo"
                ident = f"{replaced_side}a: {lead}"
                parser = _ReplayParser(
                    f"toxic-real-ko-{replaced_side}",
                    complete_prefix=True,
                    hp_visibility={"p1": "exact", "p2": "exact"},
                )
                parser.feed(
                    [
                        "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                        "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                        "|turn|1",
                        f"|-damage|{ident}|0 fnt",
                        f"|faint|{ident}",
                    ]
                )
                parser = _ReplayParser.from_snapshot(parser.snapshot())
                self.assertTrue(parser.toxic_faint_replacement_pending[replaced_side])
                parser.feed(
                    [
                        "|upkeep",
                        (
                            f"|switch|{replaced_side}a: Replacement|Replacement, L80, M|"
                            "90/100 tox"
                        ),
                        "|turn|2",
                    ]
                )
                replay = parser.snapshot()
                self.assertFalse(replay.toxic_faint_replacement_invalid[replaced_side])
                self.assertTrue(replay.toxic_stage_zero_after_upkeep[replaced_side])
                self.assertEqual(replay.toxic_stage[replaced_side], 0)
                self.assertEqual(_materialization_toxic_stage(replay, replaced_side), 0)

    def test_snapshot_toxic_proofs_require_exact_active_marker_for_both_sides(self) -> None:
        for replaced_side in ("p1", "p2"):
            with self.subTest(replaced_side=replaced_side):
                lead = "LeadOne" if replaced_side == "p1" else "LeadTwo"
                ident = f"{replaced_side}a: {lead}"
                parser = _ReplayParser(
                    f"toxic-active-marker-{replaced_side}", complete_prefix=True
                )
                parser.feed(
                    [
                        "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                        "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                        "|turn|1",
                        f"|-damage|{ident}|0 fnt",
                        f"|faint|{ident}",
                    ]
                )
                pending = parser.snapshot()
                active = pending.public_active[replaced_side]

                control = _ReplayParser.from_snapshot(pending)
                self.assertTrue(control.toxic_faint_replacement_pending[replaced_side])
                control.feed(
                    [
                        "|upkeep",
                        (
                            f"|switch|{replaced_side}a: Replacement|Replacement, L80, M|"
                            "90/100 tox"
                        ),
                        "|turn|2",
                    ]
                )
                zero_proof = control.snapshot()
                self.assertTrue(zero_proof.toxic_stage_zero_after_upkeep[replaced_side])
                self.assertEqual(_materialization_toxic_stage(zero_proof, replaced_side), 0)

                invalid_active_records = {
                    "false": replace(active, active=False),
                    "none": replace(active, active=None),
                    "truthy-int": replace(active, active=1),
                    "truthy-string": replace(active, active="yes"),
                    "malformed": object(),
                }
                for label, invalid_active in invalid_active_records.items():
                    with self.subTest(replaced_side=replaced_side, active=label):
                        restored_pending = _ReplayParser.from_snapshot(
                            replace(
                                pending,
                                public_active={
                                    **pending.public_active,
                                    replaced_side: invalid_active,
                                },
                            )
                        )
                        self.assertFalse(
                            restored_pending.toxic_faint_replacement_pending[replaced_side]
                        )

                        restored_zero = _ReplayParser.from_snapshot(
                            replace(
                                zero_proof,
                                public_active={
                                    **zero_proof.public_active,
                                    replaced_side: invalid_active,
                                },
                            )
                        ).snapshot()
                        self.assertFalse(
                            restored_zero.toxic_stage_zero_after_upkeep[replaced_side]
                        )
                        self.assertIsNone(
                            _materialization_toxic_stage(restored_zero, replaced_side)
                        )

    def test_post_upkeep_zero_requires_the_same_seat_faint(self) -> None:
        for fainted_side in (None, "p2"):
            with self.subTest(fainted_side=fainted_side):
                parser = _ReplayParser("toxic-zero-same-seat", complete_prefix=True)
                lines = [
                    "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                    "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                    "|turn|1",
                ]
                if fainted_side is not None:
                    lines.append("|faint|p2a: LeadTwo")
                lines.extend(
                    [
                        "|upkeep",
                        "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
                        "|turn|2",
                    ]
                )
                parser.feed(lines)
                replay = parser.snapshot()
                self.assertFalse(replay.toxic_stage_zero_after_upkeep["p1"])
                self.assertIsNone(_materialization_toxic_stage(replay, "p1"))

    def test_faint_latch_rejects_drag_baton_action_phase_and_duplicate_replacements(self) -> None:
        cases = {
            "drag": [
                "|upkeep",
                "|drag|p1a: Replacement|Replacement, L80, M|90/100 tox",
            ],
            "baton-pass": [
                "|upkeep",
                "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox|[from] Baton Pass",
            ],
            "action-phase-switch": [
                "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
                "|upkeep",
            ],
            "duplicate-switch": [
                "|upkeep",
                "|switch|p1a: First|First, L80, M|90/100 tox",
                "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
            ],
            "malformed-switch": [
                "|upkeep",
                "|switch|p1a: Broken",
                "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
            ],
        }
        for label, suffix in cases.items():
            with self.subTest(label=label):
                parser = _ReplayParser(f"toxic-zero-{label}", complete_prefix=True)
                parser.feed(
                    [
                        "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                        "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                        "|turn|1",
                        "|faint|p1a: LeadOne",
                        *suffix,
                        "|turn|2",
                    ]
                )
                replay = parser.snapshot()
                self.assertFalse(replay.toxic_faint_replacement_pending["p1"])
                self.assertFalse(replay.toxic_stage_zero_after_upkeep["p1"])
                self.assertIsNone(_materialization_toxic_stage(replay, "p1"))

    def test_faint_latch_rejects_malformed_order_and_forged_active_idents(self) -> None:
        cases = {
            "reversed-upkeep-faint": [
                "|upkeep",
                "|faint|p1a: LeadOne",
                "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
            ],
            "duplicate-faint": [
                "|faint|p1a: LeadOne",
                "|faint|p1a: LeadOne",
                "|upkeep",
                "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
            ],
            "duplicate-upkeep": [
                "|faint|p1a: LeadOne",
                "|upkeep",
                "|upkeep",
                "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
            ],
            "forged-active-ident": [
                "|faint|p1a: NotTheActive",
                "|upkeep",
                "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
            ],
            "unrelated-seat": [
                "|faint|p2a: LeadTwo",
                "|upkeep",
                "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
            ],
        }
        for label, suffix in cases.items():
            with self.subTest(label=label):
                parser = _ReplayParser(f"toxic-zero-malformed-{label}", complete_prefix=True)
                parser.feed(
                    [
                        "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                        "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                        "|turn|1",
                        *suffix,
                        "|turn|2",
                    ]
                )
                replay = parser.snapshot()
                self.assertFalse(replay.toxic_faint_replacement_pending["p1"])
                self.assertFalse(replay.toxic_stage_zero_after_upkeep["p1"])
                self.assertIsNone(_materialization_toxic_stage(replay, "p1"))

    def test_invalid_faint_history_is_terminal_until_a_clean_turn(self) -> None:
        cases = {
            "third-duplicate": [
                "|faint|p1a: LeadOne",
                "|faint|p1a: LeadOne",
                "|faint|p1a: LeadOne",
            ],
            "forged-then-exact": [
                "|faint|p1a: NotTheActive",
                "|faint|p1a: LeadOne",
            ],
        }
        for label, faints in cases.items():
            with self.subTest(label=label):
                parser = _ReplayParser(f"toxic-terminal-{label}", complete_prefix=True)
                parser.feed(
                    [
                        "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                        "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                        "|turn|1",
                        *faints,
                        "|upkeep",
                        "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
                        "|turn|2",
                    ]
                )
                replay = parser.snapshot()
                self.assertFalse(replay.toxic_stage_zero_after_upkeep["p1"])
                self.assertIsNone(_materialization_toxic_stage(replay, "p1"))

    def test_replacement_requires_canonical_active_ident(self) -> None:
        parser = _ReplayParser("toxic-noncanonical-replacement", complete_prefix=True)
        parser.feed(
            [
                "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                "|turn|1",
                "|faint|p1a: LeadOne",
                "|upkeep",
                "|switch|p1: Replacement|Replacement, L80, M|90/100 tox",
                "|turn|2",
            ]
        )
        replay = parser.snapshot()
        self.assertEqual(replay.public_active["p1"].ident, "p1a: LeadOne")
        self.assertFalse(replay.toxic_stage_zero_after_upkeep["p1"])

    def test_bare_active_replacement_idents_are_rejected_for_both_sides(self) -> None:
        for replaced_side in ("p1", "p2"):
            with self.subTest(replaced_side=replaced_side):
                lead = "LeadOne" if replaced_side == "p1" else "LeadTwo"
                parser = _ReplayParser(f"toxic-bare-{replaced_side}", complete_prefix=True)
                parser.feed(
                    [
                        "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                        "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                        "|turn|1",
                        f"|faint|{replaced_side}a: {lead}",
                        "|upkeep",
                        (
                            f"|switch|{replaced_side}a|Replacement, L80, M|"
                            "90/100 tox"
                        ),
                        "|turn|2",
                    ]
                )
                replay = parser.snapshot()
                self.assertEqual(
                    replay.public_active[replaced_side].ident,
                    f"{replaced_side}a: {lead}",
                )
                self.assertFalse(replay.toxic_stage_zero_after_upkeep[replaced_side])

    def test_invalid_turn_markers_retire_a_live_replacement_proof(self) -> None:
        for marker in ("|turn|2", "|turn|1", "|turn|3", "|turn|not-a-number"):
            with self.subTest(marker=marker):
                parser = _ReplayParser("toxic-invalid-turn", complete_prefix=True)
                parser.feed(
                    [
                        "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                        "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                        "|turn|1",
                        "|faint|p1a: LeadOne",
                        "|upkeep",
                        "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
                        "|turn|2",
                        marker,
                    ]
                )
                replay = parser.snapshot()
                self.assertFalse(replay.toxic_stage_zero_after_upkeep["p1"])
                self.assertIsNone(_materialization_toxic_stage(replay, "p1"))

    def test_noncanonical_expected_turn_syntax_does_not_mutate_chronology(self) -> None:
        for marker in ("|turn|2|extra", "|turn|+2", "|turn|02", " |turn|2"):
            with self.subTest(marker=marker):
                parser = _ReplayParser("toxic-turn-syntax", complete_prefix=True)
                parser.feed(
                    [
                        "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                        "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                        "|turn|1",
                        "|faint|p1a: LeadOne",
                        "|upkeep",
                        "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
                    ]
                )
                self.assertTrue(parser.toxic_stage_zero_after_upkeep["p1"])
                parser.feed([marker])
                malformed = parser.snapshot()
                self.assertEqual(malformed.turn_number, 1)
                self.assertTrue(malformed.post_upkeep_window)
                self.assertTrue(malformed.toxic_faint_replacement_invalid["p1"])
                self.assertFalse(malformed.toxic_stage_zero_after_upkeep["p1"])
                self.assertIsNone(_materialization_toxic_stage(malformed, "p1"))

                parser.feed(["|turn|2"])
                recovered = parser.snapshot()
                self.assertEqual(recovered.turn_number, 2)
                self.assertFalse(recovered.toxic_faint_replacement_invalid["p1"])
                self.assertFalse(recovered.toxic_stage_zero_after_upkeep["p1"])

    def test_snapshot_without_faint_identity_cannot_restore_or_rearm_pending_proof(self) -> None:
        parser = _ReplayParser("toxic-identity-snapshot", complete_prefix=True)
        parser.feed(
            [
                "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                "|turn|1",
                "|faint|p1a: LeadOne",
            ]
        )
        legacy = replace(
            parser.snapshot(),
            toxic_faint_replacement_expected_ident={},
            toxic_faint_replacement_invalid={},
        )
        resumed = _ReplayParser.from_snapshot(legacy)
        self.assertFalse(resumed.toxic_faint_replacement_pending["p1"])
        resumed.feed(
            [
                "|faint|p1a: LeadOne",
                "|upkeep",
                "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
                "|turn|2",
            ]
        )
        replay = resumed.snapshot()
        self.assertFalse(replay.toxic_stage_zero_after_upkeep["p1"])
        self.assertIsNone(_materialization_toxic_stage(replay, "p1"))

    def test_faint_latch_snapshot_truncation_and_scenario_reuse_fail_closed(self) -> None:
        parser = _ReplayParser("toxic-zero-truncated", complete_prefix=True)
        parser.feed(
            [
                "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                "|turn|1",
                "|faint|p1a: LeadOne",
            ]
        )
        resumed = _ReplayParser.from_snapshot(parser.snapshot())
        self.assertTrue(resumed.toxic_faint_replacement_pending["p1"])
        resumed.feed(
            [
                "|turn|2",
                "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
            ]
        )
        self.assertFalse(resumed.toxic_stage_zero_after_upkeep["p1"])

        state = {
            "turn": 2,
            "field": {"weather": "", "turnsRemaining": 0, "permanent": False},
            "sides": {
                player: {
                    "sideConditions": {},
                    "activeVolatiles": [],
                    "activeSlot": 0,
                    "pokemon": [{"status": {"id": "tox", "toxicStage": 0}}],
                }
                for player in ("p1", "p2")
            },
        }
        resumed.toxic_faint_replacement_pending["p1"] = True
        resumed.toxic_stage_zero_after_upkeep["p1"] = True
        _seed_scenario_parser_state(resumed, state)
        self.assertFalse(resumed.toxic_faint_replacement_pending["p1"])
        self.assertFalse(resumed.toxic_stage_zero_after_upkeep["p1"])

    def test_post_upkeep_drag_cannot_claim_the_replacement_zero(self) -> None:
        # In Gen 3 a drag resolves in its move action, before the residual
        # action emits |upkeep|. Treating a synthetic post-upkeep drag as a
        # faint replacement would widen the proof beyond real chronology.
        parser = _ReplayParser("toxic-post-upkeep-drag", complete_prefix=True)
        parser.feed(
            [
                "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                "|turn|1",
                "|upkeep",
                "|drag|p1a: Replacement|Replacement, L80, M|90/100 tox",
                "|turn|2",
            ]
        )
        replay = parser.snapshot()
        self.assertEqual(replay.toxic_stage["p1"], 0)
        self.assertTrue(replay.toxic_stage_known["p1"])
        self.assertFalse(replay.toxic_stage_zero_after_upkeep["p1"])
        self.assertIsNone(_materialization_toxic_stage(replay, "p1"))

    def test_post_upkeep_zero_proof_expires_with_its_first_toxic_residual(self) -> None:
        parser = _ReplayParser(
            "toxic-post-upkeep-first-tick", complete_prefix=True,
            hp_visibility={"p1": "exact", "p2": "exact"},
        )
        parser.feed(
            [
                "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                "|turn|1",
                "|faint|p1a: LeadOne",
                "|upkeep",
                "|switch|p1a: Replacement|Replacement, L80, M|100/100 tox",
                "|turn|2",
            ]
        )
        self.assertTrue(parser.toxic_stage_zero_after_upkeep["p1"])
        parser.feed(["|-damage|p1a: Replacement|94/100 tox|[from] psn"])
        replay = parser.snapshot()
        self.assertEqual(replay.toxic_stage["p1"], 1)
        self.assertFalse(replay.toxic_stage_zero_after_upkeep["p1"])
        self.assertEqual(_materialization_toxic_stage(replay, "p1"), 0)

    def test_post_upkeep_zero_proof_expires_when_its_first_tick_is_missing(self) -> None:
        cases = {
            "upkeep": ["|upkeep", "|turn|3"],
            "turn": ["|turn|3"],
        }
        for label, suffix in cases.items():
            with self.subTest(label=label):
                parser = _ReplayParser(
                    f"toxic-post-upkeep-missing-first-tick-{label}", complete_prefix=True
                )
                parser.feed(
                    [
                        "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                        "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                        "|turn|1",
                        "|faint|p1a: LeadOne",
                        "|upkeep",
                        "|switch|p1a: Replacement|Replacement, L80, M|100/100 tox",
                        "|turn|2",
                    ]
                )
                parser = _ReplayParser.from_snapshot(parser.snapshot())
                parser.feed(suffix)
                replay = parser.snapshot()
                self.assertFalse(replay.toxic_stage_zero_after_upkeep["p1"])
                self.assertIsNone(_materialization_toxic_stage(replay, "p1"))

    def test_active_toxic_zero_without_post_upkeep_replacement_proof_fails_closed(self) -> None:
        replay = SimpleNamespace(
            toxic_stage={"p1": 0},
            toxic_stage_known={"p1": True},
            toxic_stage_zero_after_upkeep={"p1": False},
            public_active={"p1": SimpleNamespace(condition="90/100 tox")},
            post_upkeep_window=False,
        )
        self.assertIsNone(_materialization_toxic_stage(replay, "p1"))

    def test_unknown_post_upkeep_toxic_zero_still_fails_closed(self) -> None:
        replay = SimpleNamespace(
            toxic_stage={"p1": 0},
            toxic_stage_known={"p1": False},
            toxic_stage_zero_after_upkeep={"p1": True},
            public_active={"p1": SimpleNamespace(condition="90/100 tox")},
            post_upkeep_window=False,
        )
        self.assertIsNone(_materialization_toxic_stage(replay, "p1"))

    def test_direct_zero_materialization_rejects_incomplete_or_tampered_provenance(self) -> None:
        parser = _ReplayParser("toxic-direct-proof", complete_prefix=True)
        parser.feed(
            [
                "|switch|p1a: LeadOne|LeadOne, L80, M|100/100",
                "|switch|p2a: LeadTwo|LeadTwo, L80, F|100/100",
                "|turn|1",
                "|-damage|p1a: LeadOne|0 fnt",
                "|faint|p1a: LeadOne",
                "|upkeep",
                "|switch|p1a: Replacement|Replacement, L80, M|90/100 tox",
                "|turn|2",
            ]
        )
        replay = parser.snapshot()
        self.assertEqual(_materialization_toxic_stage(replay, "p1"), 0)
        active = replay.public_active["p1"]
        cases = {
            "proof-not-boolean": replace(replay, toxic_stage_zero_after_upkeep={"p1": 1}),
            "proof-ident-missing": replace(replay, toxic_stage_zero_after_upkeep_ident={}),
            "proof-ident-bare": replace(
                replay, toxic_stage_zero_after_upkeep_ident={"p1": "p1a"}
            ),
            "deadline-missing": replace(
                replay, toxic_stage_zero_after_upkeep_expires_after_turn={}
            ),
            "deadline-mismatch": replace(
                replay, toxic_stage_zero_after_upkeep_expires_after_turn={"p1": 3}
            ),
            "active-ident-mismatch": replace(
                replay,
                public_active={**replay.public_active, "p1": replace(active, ident="p1a: Other")},
            ),
            "active-status-mismatch": replace(
                replay,
                public_active={
                    **replay.public_active,
                    "p1": replace(active, condition="90/100"),
                },
            ),
            "stage-unknown": replace(replay, toxic_stage_known={"p1": False}),
            "stage-nonzero": replace(replay, toxic_stage={"p1": 1}),
            "invalid-window": replace(replay, toxic_faint_replacement_invalid={"p1": True}),
            "pending-replacement": replace(
                replay, toxic_faint_replacement_pending={"p1": True}
            ),
            "stale-expected-ident": replace(
                replay, toxic_faint_replacement_expected_ident={"p1": "p1a: LeadOne"}
            ),
        }
        for label, tampered in cases.items():
            with self.subTest(label=label):
                self.assertIsNone(_materialization_toxic_stage(tampered, "p1"))

    def test_direct_materialization_requires_exact_active_marker_for_both_sides(self) -> None:
        for player in ("p1", "p2"):
            with self.subTest(player=player):
                active = ShowdownPokemon(
                    ident=f"{player}a: Replacement",
                    showdown_slot=player,
                    species="Replacement",
                    condition="90/100 tox",
                    active=True,
                )
                replay = SimpleNamespace(
                    toxic_stage={player: 0},
                    toxic_stage_known={player: True},
                    toxic_stage_zero_after_upkeep={player: True},
                    toxic_stage_zero_after_upkeep_ident={player: active.ident},
                    toxic_stage_zero_after_upkeep_expires_after_turn={player: 2},
                    toxic_faint_replacement_invalid={player: False},
                    toxic_faint_replacement_pending={player: False},
                    toxic_faint_replacement_expected_ident={player: None},
                    public_active={player: active},
                    post_upkeep_window=False,
                    turn_number=2,
                )
                self.assertEqual(_materialization_toxic_stage(replay, player), 0)

                invalid_active_records = {
                    "false": replace(active, active=False),
                    "none": replace(active, active=None),
                    "truthy-int": replace(active, active=1),
                    "truthy-string": replace(active, active="yes"),
                    "malformed": object(),
                }
                for label, invalid_active in invalid_active_records.items():
                    with self.subTest(player=player, active=label):
                        replay.public_active = {player: invalid_active}
                        self.assertIsNone(_materialization_toxic_stage(replay, player))

    def test_stage_fifteen_saturation_survives_both_boundaries(self) -> None:
        replay = SimpleNamespace(
            toxic_stage={"p1": 16},
            toxic_stage_known={"p1": True},
            public_active={"p1": SimpleNamespace(condition="1/100 tox", active=True)},
            post_upkeep_window=False,
        )

        self.assertEqual(_materialization_toxic_stage(replay, "p1"), 14)
        replay.post_upkeep_window = True
        self.assertEqual(_materialization_toxic_stage(replay, "p1"), 14)

    def test_unknown_counter_never_materializes_as_zero(self) -> None:
        replay = SimpleNamespace(
            toxic_stage={"p1": 0, "p2": 0},
            toxic_stage_known={"p1": True, "p2": False},
            public_active={
                "p1": SimpleNamespace(condition="100/100", active=True),
                "p2": SimpleNamespace(condition="85/100 tox", active=True),
            },
            post_upkeep_window=False,
        )

        self.assertIsNone(_materialization_toxic_stage(replay, "p2"))

    def test_unrepresentable_toxic_stage_fails_closed(self) -> None:
        replay = SimpleNamespace(
            toxic_stage={"p1": 17},
            toxic_stage_known={"p1": True},
            public_active={"p1": SimpleNamespace(condition="85/100 tox", active=True)},
            post_upkeep_window=False,
        )
        self.assertIsNone(_materialization_toxic_stage(replay, "p1"))

    def test_missing_provenance_blocks_only_an_active_toxic_side(self) -> None:
        replay = SimpleNamespace(
            toxic_stage={"p1": 0, "p2": 0},
            toxic_stage_known={},
            public_active={
                "p1": SimpleNamespace(condition="100/100", active=True),
                "p2": SimpleNamespace(condition="85/100 tox", active=True),
            },
            post_upkeep_window=False,
        )

        self.assertEqual(_materialization_toxic_stage(replay, "p1"), 0)
        self.assertIsNone(_materialization_toxic_stage(replay, "p2"))


class TracedAbilityParserTest(unittest.TestCase):
    """W1, parser half — the CURRENT trace only, and it dies on switch-out."""

    def _parser(self, lines: list[str]) -> _ReplayParser:
        parser = _ReplayParser("battle-gen3randombattle-1")
        parser.feed([
            "|player|p1|Us|",
            "|player|p2|Them|",
            "|switch|p1a: Gardevoir|Gardevoir, L79, M|237/237",
            "|switch|p2a: Noctowl|Noctowl, L93, F|337/337",
            *lines,
        ])
        return parser

    def test_a_trace_copy_is_recorded(self) -> None:
        p = self._parser([
            "|-ability|p1a: Gardevoir|Insomnia|Trace|[from] ability: Trace|[of] p2a: Noctowl",
        ])
        self.assertEqual(p.traced_ability["p1"], "insomnia")

    def test_a_bare_ability_reveal_is_not_a_trace(self) -> None:
        # An ordinary `|-ability|` reveal is a PERSISTENT fact about the mon's own ability
        # and belongs to the belief engine. Only the `[from] ability: Trace` tag means the
        # holder is borrowing, which is the transient case this field exists for.
        p = self._parser(["|-ability|p2a: Noctowl|Insomnia"])
        self.assertIsNone(p.traced_ability["p2"])

    def test_switching_out_drops_the_copy(self) -> None:
        # Trace re-fires on every switch-in and the copy does not survive leaving the field.
        p = self._parser([
            "|-ability|p1a: Gardevoir|Insomnia|Trace|[from] ability: Trace|[of] p2a: Noctowl",
            "|switch|p1a: Shuckle|Shuckle, L98, F|198/198",
        ])
        self.assertIsNone(p.traced_ability["p1"])

    def test_a_stale_trace_never_leaks_into_a_later_switch_in(self) -> None:
        """REGRESSION PIN for the bug this change first introduced.

        The first implementation read `belief.revealed_ability` — a PERSISTENT field — so it
        stamped the last ability the mon had ever traced. A Gardevoir that traced Levitate
        early kept being handed Levitate on every later switch-in, silently granting it
        Spikes immunity and turning a three-row fix into a two-row regression
        (seed 1500009 steps 34 and 68).

        Here the mon traces Levitate, leaves, returns, and traces Insomnia. The old code
        would answer `levitate`; the field must answer `insomnia`.
        """
        p = self._parser([
            "|-ability|p1a: Gardevoir|Levitate|Trace|[from] ability: Trace|[of] p2a: Chimecho",
            "|switch|p1a: Shuckle|Shuckle, L98, F|198/198",
            "|switch|p1a: Gardevoir|Gardevoir, L79, M|237/237",
            "|-ability|p1a: Gardevoir|Insomnia|Trace|[from] ability: Trace|[of] p2a: Noctowl",
        ])
        self.assertEqual(p.traced_ability["p1"], "insomnia")

    def test_a_trace_that_is_not_re_established_leaves_nothing_behind(self) -> None:
        # The other half of the same guard: leave and come back WITHOUT tracing again, and
        # the world must be told nothing rather than the old copy.
        p = self._parser([
            "|-ability|p1a: Gardevoir|Levitate|Trace|[from] ability: Trace|[of] p2a: Chimecho",
            "|switch|p1a: Shuckle|Shuckle, L98, F|198/198",
            "|switch|p1a: Gardevoir|Gardevoir, L79, M|237/237",
        ])
        self.assertIsNone(p.traced_ability["p1"])


class TracedAbilityPayloadTest(unittest.TestCase):
    """W1, payload half — the active row only."""

    def test_the_active_row_is_stamped(self) -> None:
        rows = [{"species": "Gardevoir", "active": True}, {"species": "Shuckle", "active": False}]
        _apply_traced_ability_materialization_state(rows, "insomnia")
        self.assertEqual(rows[0]["revealedAbility"], "insomnia")
        self.assertNotIn("revealedAbility", rows[1])

    def test_no_trace_is_a_no_op(self) -> None:
        rows = [{"species": "Gardevoir", "active": True}]
        for traced in (None, ""):
            _apply_traced_ability_materialization_state(rows, traced)
            self.assertNotIn("revealedAbility", rows[0])


class TracedAbilityWorldTest(unittest.TestCase):
    """W1, world half — the confirmed ability wins, and nothing else changes."""

    def test_a_confirmed_ability_beats_the_sampled_one(self) -> None:
        from pokezero.engine_world import _resolved_ability

        class _Mon:
            ability = "Trace"

        self.assertEqual(_resolved_ability(_Mon(), {"revealedAbility": "flashfire"}), "flashfire")

    def test_the_sampled_ability_is_used_when_nothing_was_traced(self) -> None:
        from pokezero.engine_world import _resolved_ability

        class _Mon:
            ability = "Trace"

        for row in (None, {}, {"revealedAbility": ""}, {"revealedAbility": "   "}):
            self.assertEqual(
                _resolved_ability(_Mon(), row), "trace", f"row {row!r} should fall through"
            )

    def test_a_mon_with_no_sampled_ability_stays_none(self) -> None:
        from pokezero.engine_world import _resolved_ability

        class _Mon:
            ability = None

        self.assertIsNone(_resolved_ability(_Mon(), None))

    def test_only_the_ability_field_is_seeded_no_activation(self) -> None:
        """gen3 does not fire the copied ability's Start event on acquisition (#962 patch 32).

        `_resolved_ability` returns a bare ability id and nothing else — no volatile, no
        instruction. Pinned as a shape assertion because "seed the ability AND its
        on-switch-in effect" is the plausible over-implementation, and in gen3 it is wrong.
        """
        from pokezero.engine_world import _resolved_ability

        class _Mon:
            ability = "Trace"

        self.assertIsInstance(_resolved_ability(_Mon(), {"revealedAbility": "flashfire"}), str)


if __name__ == "__main__":
    unittest.main()
