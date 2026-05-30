import inspect
import unittest
from typing import get_type_hints

from pokezero.env import AsyncPokeZeroEnv, PokeZeroEnv, StepResult, TerminalState
from pokezero.observation import ObservationSpec, PokeZeroObservationV0


def _observation() -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=2, numeric_feature_count=3)
    return PokeZeroObservationV0(
        categorical_ids=tuple((0, 0) for _ in range(spec.token_count)),
        numeric_features=tuple((0.0, 0.0, 0.0) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=(True, False, False, False, False, False, False, False, False),
    )


class DummyEnv:
    def __init__(self) -> None:
        self._observation = _observation()
        self._terminal = None
        self._requested_players = ("p1", "p2")

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self._terminal = None

    def observe(self, player: str) -> PokeZeroObservationV0:
        return self._observation

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self._observation.legal_action_mask

    def requested_players(self) -> tuple[str, ...]:
        return self._requested_players

    def step(self, actions: dict[str, int]) -> StepResult:
        if tuple(actions) != self._requested_players:
            raise ValueError("actions must be provided for the currently requested players.")
        self._terminal = TerminalState(winner="p1", turn_count=1)
        self._requested_players = ()
        return StepResult(
            observations={"p1": self._observation, "p2": self._observation},
            rewards={"p1": 1.0, "p2": -1.0},
            terminal=self._terminal,
            requested_players=self._requested_players,
        )

    def terminal(self) -> TerminalState | None:
        return self._terminal


class DummyAsyncEnv:
    def __init__(self) -> None:
        self._env = DummyEnv()

    async def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self._env.reset(seed=seed, format_id=format_id)

    async def observe(self, player: str) -> PokeZeroObservationV0:
        return self._env.observe(player)

    async def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self._env.legal_actions(player)

    async def requested_players(self) -> tuple[str, ...]:
        return self._env.requested_players()

    async def step(self, actions: dict[str, int]) -> StepResult:
        return self._env.step(actions)

    async def terminal(self) -> TerminalState | None:
        return self._env.terminal()


class EnvProtocolTest(unittest.TestCase):
    def test_dummy_env_smoke_test_for_protocol_methods(self) -> None:
        env = DummyEnv()

        self.assertIsInstance(env, PokeZeroEnv)
        env.reset(seed=123)
        self.assertEqual(env.requested_players(), ("p1", "p2"))
        self.assertEqual(env.legal_actions("p1"), (True, False, False, False, False, False, False, False, False))
        result = env.step({"p1": 0, "p2": 0})
        self.assertEqual(result.terminal, TerminalState(winner="p1", turn_count=1))
        self.assertEqual(result.requested_players, ())

    def test_step_accepts_only_currently_requested_players(self) -> None:
        env = DummyEnv()
        env._requested_players = ("p1",)

        result = env.step({"p1": 4})

        self.assertEqual(result.terminal, TerminalState(winner="p1", turn_count=1))

    def test_step_rejects_missing_or_extra_players(self) -> None:
        env = DummyEnv()
        env._requested_players = ("p1",)

        with self.assertRaisesRegex(ValueError, "currently requested"):
            env.step({"p1": 4, "p2": 0})

    def test_sync_protocol_exposes_expected_signatures(self) -> None:
        step_signature = inspect.signature(PokeZeroEnv.step)
        hints = get_type_hints(PokeZeroEnv.step)

        self.assertIn("actions", step_signature.parameters)
        self.assertEqual(hints["return"], StepResult)

    def test_async_protocol_is_available_for_event_loop_backends(self) -> None:
        env = DummyAsyncEnv()

        self.assertIsInstance(env, AsyncPokeZeroEnv)
        self.assertTrue(inspect.iscoroutinefunction(AsyncPokeZeroEnv.step))


if __name__ == "__main__":
    unittest.main()
