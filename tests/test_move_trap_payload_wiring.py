"""The Mean Look / Spider Web move trap must REACH the world builder.

`engine_world` grew a `trapped` volatile and an exemption in
`_require_world_reproduces_trap` so a move-trapped seat could be searched instead of refused.
Both were unreachable in production: the only signal the parser produced was
`ShowdownReplayState.meanlook_trap`, an OBSERVATION-lane tracker, while the world builder reads
`sides[slot]["volatiles"]`, which is `_update_volatiles`'s output and gated on TRACKED_VOLATILES
-- a set `trapped` is deliberately not in. So `sides[self].volatiles` was `[]` on every Mean Look
turn, the exemption could not fire, and the decision landed on

    self_request_state_unsupported: self active request flags ['trapped'] constrain legality
    beyond this construction (sampled world does not trap: foe ability 'X')

whose foe ability is a bystander -- the same decision names a different one on each retry,
because it is whichever ability the belief sample happened to draw.

WHY A NEW PAYLOAD KEY RATHER THAN A `volatiles` ENTRY. The two lanes exist for a reason:
`volatiles` is TRACKED_VOLATILES-gated, and that set is simultaneously (a) the observation
encoder's `volatile:<id>` vocabulary, where this fact ALREADY lives as the v3 numeric column
NUMERIC_MEANLOOK_TRAP, and (b) mirrored by the Node bridge's STATIC_PUBLIC_VOLATILES, which
throws on anything else and could not rebuild the linked `trapper` volatile anyway. Adding
`trapped` there would unfreeze v3 tensors and break `materialize`. A dedicated key is the
established shape for a parser fact only the world lane consumes -- `truantPhase` and
`stallCounter` are already exactly that.

The seam test below is the one that matters: it drives real protocol lines through the
production parser, builds the payload with the production builder, and feeds THAT payload --
not a hand-written literal -- to the world constructor. A rename on either side of the key
breaks it, because the test never spells the key.
"""

from __future__ import annotations

import unittest

from pokezero.engine_world import EngineWorldUnsupported, battle_spec_from_payload
from pokezero.showdown import parse_showdown_replay

from ._showdown_root import requires_showdown
from .test_engine_world import _dex, _move_trap_support, _override, _payload

_MEAN_LOOK = (
    "|move|p2a: Snorlax|Mean Look|p1a: Swampert",
    "|-activate|p1a: Swampert|trapped",
)


def _lines(*extra: str) -> list[str]:
    """A minimal public transcript seated at p1, species matching `_payload`/`_override`."""

    return [
        "|player|p1|Alice|1|",
        "|player|p2|Bob|2|",
        "|teamsize|p1|2",
        "|teamsize|p2|2",
        "|gen|3",
        "|tier|[Gen 3] Random Battle",
        "|start",
        "|switch|p1a: Swampert|Swampert, L84, M|100/100",
        "|switch|p2a: Snorlax|Snorlax, L80, M|100/100",
        "|turn|1",
        *extra,
        "|turn|2",
    ]


# `_request_materialization_rows` needs a real acting-player request; an empty stub raises.
_SELF_REQUEST = {
    "side": {
        "pokemon": [
            {
                "ident": "p1: Swampert",
                "details": "Swampert, L84, M",
                "condition": "100/100",
                "active": True,
                "moves": ["earthquake", "icebeam"],
            },
            {
                "ident": "p1: Starmie",
                "details": "Starmie, L79, M",
                "condition": "100/100",
                "active": False,
                "moves": ["surf"],
            },
        ]
    }
}

# The keys the hand-built `_payload` fixture owns: exact self HP, request-known move PP, the
# sampled boosts. Everything else on a produced side is spliced from the PRODUCER below, which
# is what makes the join real -- the test never names the move-trap key, so renaming it on
# either side of the seam fails these tests instead of silently passing.
_FIXTURE_OWNED_SIDE_KEYS = frozenset(
    {
        "pokemon",
        "boosts",
        "volatiles",
        "materializationBlockers",
        "toxicStage",
        "sideConditions",
        "sideConditionSetTurns",
    }
)


@requires_showdown()
class MoveTrapReachesTheWorldBuilderTest(unittest.TestCase):
    """Producer -> payload -> consumer, with no hand-written key on the join."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dex = _dex()

    def _produced_payload(self, *extra: str) -> dict:
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.local_showdown import (
            PublicBattleMaterializationState,
            _public_materialization_payload,
        )

        replay = parse_showdown_replay(
            _lines(*extra), battle_id="battle-move-trap-seam-1", complete_prefix=True
        )
        state = PublicBattleMaterializationState(
            player_id="p1",
            format_id="gen3randombattle",
            observation_format_id="gen3randombattle",
            replay=replay,
            belief_engine=PublicBattleBeliefEngine.from_events(
                replay.public_events, format_id="gen3randombattle"
            ),
            self_request=_SELF_REQUEST,
            self_move_states={},
            self_initial_request=_SELF_REQUEST,
        )
        return _public_materialization_payload(state)

    def _spliced_payload(self, *extra: str) -> dict:
        """`_payload`'s constructible world, carrying the PRODUCER's own side facts."""

        produced = self._produced_payload(*extra)
        payload = _payload(self.dex)
        # Showdown discloses `trapped: true` on a move-trapped seat's request; that flag is
        # what `_require_world_reproduces_trap` has to discharge.
        payload["selfActiveRequestState"] = {"trapped": True}
        for slot in ("p1", "p2"):
            for key, value in produced["sides"][slot].items():
                if key not in _FIXTURE_OWNED_SIDE_KEYS:
                    payload["sides"][slot][key] = value
        return payload

    def test_the_parsers_move_trap_reaches_the_built_world(self) -> None:
        payload = self._spliced_payload(*_MEAN_LOOK)
        with _move_trap_support(lambda engine=None: None):
            world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertIn(
            "trapped",
            world.spec.side_one.volatile_statuses,
            "the parser saw |-activate|p1a: Swampert|trapped but the world was built free -- "
            "the payload lane never carried the move trap",
        )

    def test_the_disclosed_request_flag_is_discharged_rather_than_refused(self) -> None:
        """The whole class. Same payload, and the decision is searched instead of declined."""

        payload = self._spliced_payload(*_MEAN_LOOK)
        with _move_trap_support(lambda engine=None: None):
            battle_spec_from_payload(payload, _override(), dex=self.dex)

    def test_an_untrapped_transcript_still_fails_closed(self) -> None:
        """The control, and the reason the test above is not vacuous.

        Identical fixture, identical disclosed request flag, only the two Mean Look lines
        removed. The foe here has ability Immunity -- no Shadow Tag, no Arena Trap, no Magnet
        Pull -- so nothing in the sampled world traps us and the refusal is still correct.
        """

        payload = self._spliced_payload()
        with _move_trap_support(lambda engine=None: None):
            with self.assertRaises(EngineWorldUnsupported) as caught:
                battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertEqual(caught.exception.reason, "self_request_state_unsupported")

    def test_the_trap_follows_the_trapped_seat_and_not_the_trapper(self) -> None:
        """Non-vacuity for the per-slot read: `always set it on p1` would pass the tests above.

        Here p2 is the one Mean Looked, so the volatile must land on side_two and side_one must
        be free -- which also means the disclosed self flag is NOT discharged and the fixture
        still refuses, so the assertion is on the raise path.
        """

        payload = self._spliced_payload(
            "|move|p1a: Swampert|Mean Look|p2a: Snorlax",
            "|-activate|p2a: Snorlax|trapped",
        )
        payload["selfActiveRequestState"] = {}
        with _move_trap_support(lambda engine=None: None):
            world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertIn("trapped", world.spec.side_two.volatile_statuses)
        self.assertNotIn("trapped", world.spec.side_one.volatile_statuses)


@requires_showdown()
class MoveTrapPayloadConsumerTest(unittest.TestCase):
    """The consumer half in isolation, including the stale-wheel gate."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dex = _dex()

    def test_the_key_is_honoured_only_when_it_is_true(self) -> None:
        payload = _payload(self.dex)
        payload["sides"]["p1"]["meanlookTrap"] = False
        with _move_trap_support(lambda engine=None: None):
            world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertNotIn("trapped", world.spec.side_one.volatile_statuses)

    def test_the_key_takes_the_same_wheel_gate_a_payload_volatile_takes(self) -> None:
        """A stale wheel drops TRAPPED silently, so the joined-in volatile must fail closed too.

        Without this the new producer would be a way AROUND `require_move_trap_support`: search
        would hand the trapped seat its switch options back on an unpatched wheel, which is
        strictly worse than the refusal this change removes.
        """

        from pokezero.poke_engine_adapter import PokeEngineMoveTrapUnsupportedError

        def _stale(engine=None):
            raise PokeEngineMoveTrapUnsupportedError("no move-trapping.patch")

        payload = _payload(self.dex)
        payload["sides"]["p1"]["meanlookTrap"] = True
        with _move_trap_support(_stale):
            with self.assertRaises(PokeEngineMoveTrapUnsupportedError):
                battle_spec_from_payload(payload, _override(), dex=self.dex)

    def test_a_payload_without_the_key_is_unchanged(self) -> None:
        """Strictly additive: an older cached payload builds exactly as before."""

        payload = _payload(self.dex)
        self.assertNotIn("meanlookTrap", payload["sides"]["p1"])
        world = battle_spec_from_payload(payload, _override(), dex=self.dex)
        self.assertNotIn("trapped", world.spec.side_one.volatile_statuses)


class StaleWheelIsAFallbackNotACrashTest(unittest.TestCase):
    """`require_move_trap_support`'s raise is newly REACHABLE, so `_search` must attribute it.

    The guard has existed since the move-trap patch landed, but nothing in production ever put
    a TRAPPED volatile in a payload, so no live caller could reach it. Routing the parser's
    move trap into the payload makes it live -- and `PokeEngineMoveTrapUnsupportedError` is a
    `PokeEngineUnavailableError`, not an `EngineWorldUnsupported`, so the existing handler does
    not catch it. Unhandled, an unpatched wheel would turn every Mean Look turn from a
    fallback into a crashed run.

    Mirrors `tests/test_engine_search.py::AttractPatchFallbackTests`, which pins the same shape
    for the Attract probe.
    """

    def test_a_missing_move_trap_patch_is_an_attributed_fallback(self) -> None:
        import random
        from collections import Counter
        from unittest import mock

        from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy
        from pokezero.engine_world import EngineWorld
        from pokezero.poke_engine_adapter import PokeEngineMoveTrapUnsupportedError

        from .test_engine_search import _FakeContext, _FakeObservation, _candidates

        policy = EngineMctsPolicy(
            dex=None,
            set_source=None,
            module=mock.Mock(),
            config=EngineMctsConfig(worlds=1, sample_retry_factor=1),
        )
        mask = (True, False, False, False, False, False, False, False, False)
        context = _FakeContext(_FakeObservation(mask, _candidates()))
        with mock.patch(
            "pokezero.engine_search._gen3_randbat_belief_start_override_result",
            return_value=(object(), None),
        ), mock.patch(
            "pokezero.engine_search.world_battle_spec",
            side_effect=PokeEngineMoveTrapUnsupportedError("no move-trapping.patch"),
        ):
            decision = policy.select_action_with_context(context, rng=random.Random(0))

        self.assertEqual(decision.metadata["engine_mcts"]["fallback"], "no_worlds_constructed")
        self.assertEqual(
            policy.stats.world_failure_reasons,
            Counter({"move_trap_patch_unavailable": 1}),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
