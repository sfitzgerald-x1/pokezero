"""Local authoring and evaluation tools for Gen 3 randbats endgame scenarios."""

from .domain import ENDGAME_SCENARIO_SCHEMA_VERSION, EndgameScenario, ScenarioValidationError

__all__ = (
    "ENDGAME_SCENARIO_SCHEMA_VERSION",
    "EndgameScenario",
    "ScenarioValidationError",
)
