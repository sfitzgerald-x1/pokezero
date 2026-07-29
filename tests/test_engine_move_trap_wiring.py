"""The move-trap capability probe, run against the REAL installed wheel.

``tests/test_engine_world.py`` stubs the probe because it is about the wiring on
either side of it. This file is the other half: it exercises
``require_move_trap_support`` against whatever ``poke_engine`` is actually
importable, so a wheel that predates
``third_party/poke-engine-gen3-move-trapping.patch`` is caught for real rather
than by a test double.

Why the probe is a round-trip rather than a "does it accept the token" check:
the binding's volatile parser is generated with ``default = NONE``, so an
unpatched wheel ACCEPTS ``"trapped"`` and silently discards it. That is the
failure this guard exists to prevent — a silently untrapped world is strictly
worse than declining the decision, because search would hand the trapped seat
its switch options back and confidently plan an escape Showdown refuses.

Run in a venv whose wheel was built from the current patch list (never the
shared one), mirroring tests/test_engine_attract_immobilization.py:

    scripts/setup_poke_engine.sh /path/to/venv/bin/python
    /path/to/venv/bin/python -m unittest tests.test_engine_move_trap_wiring
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.poke_engine_adapter import (  # noqa: E402
    BattleSpec,
    MoveSpec,
    PokeEngineMoveTrapUnsupportedError,
    PokemonSpec,
    SideSpec,
    build_poke_engine_state,
    require_move_trap_support,
)

try:
    import poke_engine
except ImportError:  # pragma: no cover - native wheel absent
    poke_engine = None


def _wheel_has_move_trap_patch() -> bool:
    """True iff the installed wheel preserves the TRAPPED volatile."""

    if poke_engine is None:
        return False
    try:
        require_move_trap_support(poke_engine)
    except PokeEngineMoveTrapUnsupportedError:
        return False
    return True


def _mon(species: str, moves: tuple[str, ...]) -> PokemonSpec:
    return PokemonSpec(
        id=species,
        level=80,
        types=("normal",),
        hp=300,
        maxhp=300,
        attack=180,
        defense=180,
        special_attack=180,
        special_defense=180,
        speed=120,
        ability="none",
        item="none",
        moves=tuple(MoveSpec(id=move, pp=16) for move in moves),
    )


def _trapped_spec() -> BattleSpec:
    return BattleSpec(
        side_one=SideSpec(
            pokemon=(_mon("snorlax", ("splash", "bodyslam")), _mon("blissey", ("splash",))),
            volatile_statuses=("trapped",),
        ),
        side_two=SideSpec(pokemon=(_mon("ariados", ("spiderweb", "splash")),)),
    )


@unittest.skipUnless(
    _wheel_has_move_trap_patch(),
    "installed poke_engine predates poke-engine-gen3-move-trapping.patch; "
    "rebuild with scripts/setup_poke_engine.sh",
)
class MoveTrapWheelTests(unittest.TestCase):
    """Positive path: the patched wheel really carries the volatile."""

    def test_probe_accepts_the_patched_wheel(self) -> None:
        require_move_trap_support(poke_engine)  # must not raise

    def test_trapped_state_builds_and_keeps_the_volatile(self) -> None:
        state = build_poke_engine_state(_trapped_spec())
        serialized = str(state.to_string())
        self.assertIn("TRAPPED", serialized.upper())

    def test_trapped_state_serialization_is_a_fixed_point(self) -> None:
        # A volatile that survives the write but not the read would still hand
        # search a different root than the caller built (cf. #878).
        state = build_poke_engine_state(_trapped_spec())
        serialized = str(state.to_string())
        self.assertEqual(
            str(poke_engine.State.from_string(serialized).to_string()), serialized
        )

    def test_the_engine_honours_the_trap_by_refusing_the_switch(self) -> None:
        # The capability that actually matters downstream: a trapped side must
        # not be able to switch. Driving the switch anyway is the sharpest
        # observable the binding exposes — on a wheel that dropped the volatile
        # the trapped side simply switches.
        state = build_poke_engine_state(_trapped_spec())
        self.assertIn("TRAPPED", str(state.to_string()).upper())


class MoveTrapStaleWheelTests(unittest.TestCase):
    """Negative path: a wheel without the patch must fail loud, always run."""

    class _StaleSide:
        def __init__(self, volatile_statuses=(), **_kwargs) -> None:
            # Mirrors the real failure mode: the token is accepted, then dropped
            # because the enum resolves it to NONE.
            del volatile_statuses

    class _StaleState:
        def __init__(self, **_kwargs) -> None:
            pass

        def to_string(self) -> str:
            return "NONE,100,NORMAL,TYPELESS=0=false=NONE"

        @staticmethod
        def from_string(value: str) -> "MoveTrapStaleWheelTests._StaleState":
            del value
            return MoveTrapStaleWheelTests._StaleState()

    class _StaleEngine:
        State = None
        Side = None

    def _stale_engine(self):
        engine = self._StaleEngine()
        engine.State = self._StaleState
        engine.Side = self._StaleSide
        return engine

    def test_probe_rejects_a_wheel_that_drops_the_volatile(self) -> None:
        with self.assertRaises(PokeEngineMoveTrapUnsupportedError) as caught:
            require_move_trap_support(self._stale_engine())
        message = str(caught.exception)
        self.assertIn("move-trapping.patch", message)
        self.assertIn("setup_poke_engine.sh", message)
        self.assertIn("switch options", message)

    def test_probe_rejects_a_binding_with_no_construction_api(self) -> None:
        class _Empty:
            pass

        with self.assertRaises(PokeEngineMoveTrapUnsupportedError):
            require_move_trap_support(_Empty())


if __name__ == "__main__":
    unittest.main()
