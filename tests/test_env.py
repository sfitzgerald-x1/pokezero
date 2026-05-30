import unittest

from pokezero.env import PokeZeroEnv, StepResult, TerminalState
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

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self._terminal = None

    def observe(self, player: str) -> PokeZeroObservationV0:
        return self._observation

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self._observation.legal_action_mask

    def step(self, actions: dict[str, int]) -> StepResult:
        self._terminal = TerminalState(winner="p1", turn_count=1)
        return StepResult(
            observations={"p1": self._observation, "p2": self._observation},
            rewards={"p1": 1.0, "p2": -1.0},
            terminal=self._terminal,
        )

    def terminal(self) -> TerminalState | None:
        return self._terminal


class EnvProtocolTest(unittest.TestCase):
    def test_dummy_env_satisfies_protocol_shape(self) -> None:
        env = DummyEnv()

        self.assertIsInstance(env, PokeZeroEnv)
        env.reset(seed=123)
        self.assertEqual(env.legal_actions("p1"), (True, False, False, False, False, False, False, False, False))
        result = env.step({"p1": 0, "p2": 0})
        self.assertEqual(result.terminal, TerminalState(winner="p1", turn_count=1))


if __name__ == "__main__":
    unittest.main()
