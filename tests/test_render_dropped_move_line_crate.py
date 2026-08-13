"""The dropped-`|move|`-line instrument, driven by the SHIPPED renderer.

`tests/test_public_projection.DroppedMoveLineTests` pins the comparator against
hand-supplied branch payloads; that is the right level for the null worlds, and
it is also a level at which a renderer change cannot turn any of it red. This
file closes that gap: every branch below comes out of
`pokezero_search.branch_events` on a constructed gen3 state, so a renderer that
stops dropping the line — or starts dropping another one — moves these tests.

WHAT IS AND IS NOT A SHOWDOWN LOG HERE. `observed_lines` is written out rather
than captured from the sim. It has to be: reaching this position in a real
`gen3randombattle` needs a Rest-asleep Sleep Talk user that also carries a
party-wide cure, which the 256-game control block never produced. The lines
below are therefore a REFERENCE, not a capture, and their only load-bearing
content is the `|move|` announcement multiset — `fold_step_lines` has no `move`
arm, so nothing else in them reaches the comparison. Evidence from real logs is
the control-block run recorded in the PR body, not this file.

THE POSITION. gen3's party-wide cures walk `pokemon_index_iter()` in SLOT ORDER
(`gen3/choice_effects.rs`, `Choices::HEALBELL`), so which party member is
statused, and at which slot, decides whether the cure's FIRST instruction lands
on the active. #1242 guarded `consume_move_prelude`'s wake and thaw arms on the
active slot, which fixed the layouts where a BENCHED clear leads. It explicitly
did not fix — and pins as a known remaining gap in
`rust/pokezero-search/tests/gen3_sleeptalk_party_cure_active_slot_guard.rs` —
the layouts where the ACTIVE's OWN clear leads, because there the guard cannot
separate that clear from a genuine wake. Both are exercised below, and they are
the two directions this instrument has to tell apart.
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import pokezero_search
except ImportError:  # pragma: no cover - native module optional in some venvs
    pokezero_search = None

from pokezero.poke_engine_adapter import (  # noqa: E402
    BattleSpec,
    MoveSpec,
    PokemonSpec,
    SideSpec,
    build_poke_engine_state,
)
from pokezero.public_projection import render_projection_mismatch  # noqa: E402

SLOT_SIDES = {"p1": "side_one", "p2": "side_two"}
CTX = json.dumps({"p1": ["Bench", "Sleeper"], "p2": ["Opponent"], "turn": 1})

#: The announcement the render must carry once the callee is identified. Sleep
#: Talk's own line is separate; THIS is the called move, and it is the line the
#: mis-cut prelude loses.
CALLEE = "|move|p1a|healbell"

#: The reference log for this turn. `|-cureteam|` is deliberately absent: the
#: renderer emits no protocol line for a fold-ignored cure, so including it would
#: make the boundary fold UNEQUAL on status and the comparator would report
#: `render_unmatched_transition` before the announcement check ever ran. Leaving
#: it out isolates the axis under test, and is stated here rather than left for a
#: reader to infer from a passing test.
OBSERVED = [
    "|cant|p1a: Sleeper|slp",
    "|move|p1a: Sleeper|Sleep Talk|p1a: Sleeper",
    "|move|p1a: Sleeper|Heal Bell|p1a: Sleeper|[from]Sleep Talk",
    "|move|p2a: Opponent|Splash||[still]",
    "|upkeep",
    "|turn|2",
]

#: Both actives sit at full HP with no status by the end of the turn, which is
#: what makes the drop silent: every axis the fold compares agrees.
PRE = types.SimpleNamespace(
    p1_hp=300,
    p2_hp=300,
    p1_status="NONE",
    p2_status="NONE",
    fainted=frozenset(),
    weather="NONE",
    side_conditions={},
    presence=lambda: {},
)


def _mon(
    species: str,
    moves: tuple[str, ...],
    *,
    speed: int,
    status: str = "none",
    rest_turns: int = 0,
) -> PokemonSpec:
    return PokemonSpec(
        id=species,
        level=100,
        types=("normal",),
        hp=300,
        maxhp=300,
        attack=100,
        defense=100,
        special_attack=100,
        special_defense=100,
        speed=speed,
        status=status,
        ability=None,
        item=None,
        rest_turns=rest_turns,
        moves=tuple(MoveSpec(id=move, pp=32) for move in moves),
    )


def _sleeper() -> PokemonSpec:
    """Rest-asleep with two turns left, so the sleep gate fires and the mon does
    NOT wake this turn — the callee's instructions, not a wake, are what follows.
    Carries the party-wide cure among its Sleep Talk callees."""

    return _mon(
        "snorlax",
        ("sleeptalk", "healbell", "rest"),
        speed=500,
        status="sleep",
        rest_turns=2,
    )


@unittest.skipIf(pokezero_search is None, "pokezero_search native module not built")
class DroppedCalleeLineThroughTheCrateTests(unittest.TestCase):
    def _run(self, side_one: SideSpec):
        spec = BattleSpec(
            side_one=side_one,
            side_two=SideSpec(pokemon=(_mon("chansey", ("splash",), speed=1),)),
        )
        try:
            state = build_poke_engine_state(spec).to_string()
        except Exception as exc:  # pragma: no cover - unpatched/absent wheel
            self.skipTest(f"poke_engine fixture unavailable: {exc}")
        return render_projection_mismatch(
            state_string=state,
            slot_sides=SLOT_SIDES,
            party_display={"p1": ["Bench", "Sleeper"], "p2": ["Opponent"]},
            turn=1,
            choices={"p1": "sleeptalk", "p2": "splash"},
            observed_lines=OBSERVED,
            pre_features=PRE,
        )

    def test_the_layout_1242_fixed_renders_the_callee_and_the_axis_is_silent(self):
        """THE CONTROL, and it is the post-fix render of a REAL defect.

        A benched party member below the active puts a NON-active clear at the
        head of the callee's instructions. Before #1242 the prelude ate it and
        the render lost the callee line; after #1242 it does not, and the
        instrument agrees by staying quiet. Without this test the firing test
        below could be passing because the axis fires on every Sleep Talk turn.
        """

        found, diagnostics = self._run(
            SideSpec(
                pokemon=(
                    _mon("snorlax", ("splash",), speed=500, status="freeze"),
                    _sleeper(),
                ),
                active_index=1,
            )
        )
        self.assertEqual([], [m.axis for m in found], diagnostics)
        self.assertEqual([], diagnostics["move_lines_dropped"])
        # THE DENOMINATOR. Three announcements were compared, not zero: a silent
        # axis on a boundary where the check never ran would prove nothing.
        self.assertEqual(3, diagnostics["move_lines_compared"])

    def test_the_known_remaining_gap_is_now_VISIBLE(self):
        """MUST FIRE, on `main`, unforced, TODAY.

        With no statused bench the cure's walk reaches the ACTIVE's own clear
        first, the wake arm consumes it, `still_asleep` goes false, and
        `render_move_phase` exits through its "awake Sleep Talk always fails"
        arm — emitting Sleep Talk's own line and nothing else. `lossy` and
        `attribution_unsafe` are both empty, which is the whole problem: no
        refusal, no marker, no direction-1 row. This is the layout
        `the_active_first_walk_is_a_known_remaining_gap_with_no_bench` pins as
        UNFIXED, and it is the first time the gap is counted rather than
        described.

        This test is therefore also the strongest available evidence that the
        instrument is not fitted to a zero: the defect it detects predates it and
        is still live in the tree that ships it.

        NULL WORLD: `dropped_move_lines` returning `[]`.
        """

        found, diagnostics = self._run(
            SideSpec(pokemon=(_sleeper(),), active_index=0)
        )
        self.assertEqual(
            ["render_move_line_dropped"], [m.axis for m in found], diagnostics
        )
        self.assertEqual([CALLEE], diagnostics["move_lines_dropped"])
        self.assertEqual(
            f"render_move_line_dropped:{CALLEE}", found[0].predicate
        )

    def test_the_higher_slot_bench_layout_of_the_same_gap_also_fires(self):
        """The SECOND layout of the same gap, which an earlier revision of the
        Rust fixture's note missed: a statused bench at a HIGHER slot than the
        active is walked AFTER it, so the active's own clear still leads. Pinning
        one of the two invites the reader to believe the other is fixed."""

        found, diagnostics = self._run(
            SideSpec(
                pokemon=(
                    _sleeper(),
                    _mon("snorlax", ("splash",), speed=500, status="freeze"),
                ),
                active_index=0,
            )
        )
        self.assertEqual(
            ["render_move_line_dropped"], [m.axis for m in found], diagnostics
        )
        self.assertEqual([CALLEE], diagnostics["move_lines_dropped"])


class RenderForcingTests(unittest.TestCase):
    """The forcing mode, which is the other half of the instrument.

    A counter that reads 0 is worthless unless it can be made to read non-zero.
    `--force-render render-drop-move-line` re-creates the dropped-line condition
    on the model of direction 1's `--force abort`: it makes the pipeline report
    something it would otherwise be silent about, which is what makes a zero from
    the unforced arm admissible as evidence rather than an absence of
    measurement.

    Tested here rather than only at census scale because the census run is
    minutes and this is milliseconds, and because a forcing that silently no-ops
    is the exact failure the direction-1 apparatus already recorded: its first
    revision poked immutable pyo3 objects, swallowed 312 `AttributeError`s per
    game, and reported a clean zero while doing so.
    """

    def setUp(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import public_projection_census  # noqa: PLC0415

        self.module = public_projection_census

    def test_the_forcing_deletes_exactly_one_move_line_per_branch(self):
        class _Crate:
            def branch_events(self, *_args, **_kwargs):
                return json.dumps(
                    {
                        "branches": [
                            {
                                "events": [
                                    "|move|p1a: X|sleeptalk|p1a: X",
                                    "|move|p1a: X|healbell|p1a: X|[from] Sleep Talk",
                                    "|turn|4",
                                ],
                                "lossy": [],
                            }
                        ]
                    }
                )

        forced = self.module.RENDER_FORCINGS["render-drop-move-line"](_Crate())
        report = json.loads(forced.branch_events("s", "a", "b", "{}", True, False))
        # THE LAST one, so the callee line is what goes and the damage that
        # follows keeps the same owner.
        self.assertEqual(
            [
                "|move|p1a: X|sleeptalk|p1a: X",
                "|turn|4",
            ],
            report["branches"][0]["events"],
        )
        self.assertEqual(1, forced.dropped)

    def test_the_forcing_leaves_a_branch_with_no_move_line_untouched(self):
        """It must not fabricate a deletion where there is nothing to delete: a
        forcing that changes boundaries it was not aiming at would light up axes
        it does not target, and then a lit axis proves nothing."""

        class _Crate:
            def branch_events(self, *_args, **_kwargs):
                return json.dumps(
                    {"branches": [{"events": ["|cant|p1a: X|slp", "|turn|4"]}]}
                )

        forced = self.module.RENDER_FORCINGS["render-drop-move-line"](_Crate())
        report = json.loads(forced.branch_events("s", "a", "b", "{}", True, False))
        self.assertEqual(["|cant|p1a: X|slp", "|turn|4"], report["branches"][0]["events"])
        self.assertEqual(0, forced.dropped)

    def test_an_unknown_forcing_mode_raises_instead_of_degrading_to_none(self):
        """A typo that silently disarms the instrument test is the shape of a
        harness that reports success while measuring nothing. Same rule the
        world-forcing registry already enforces."""

        self.assertIsNone(self.module.resolve_render_forcing("none"))
        with self.assertRaises(SystemExit):
            self.module.resolve_render_forcing("render-drop-move-lines")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
