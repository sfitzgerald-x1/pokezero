"""Application service shared by the local HTTP API, tests, and future batch tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..checkpoint_factors import choice_label
from ..local_showdown import LocalShowdownConfig, LocalShowdownEnv, env_config_with_checkpoint_masks
from ..neural_policy import (
    evaluate_transformer_action_priors,
    feature_masks_from_model_config,
    load_transformer_checkpoint,
    observation_spec_from_model_config,
)
from .catalog import ScenarioCatalog, scenario_bridge_patch, scenario_start_override, validate_scenario
from .domain import EndgameScenario
from .storage import ScenarioRepository


class ScenarioStudioService:
    """Own the local source catalog, Showdown validation, and JSON scenario repository."""

    def __init__(self, *, showdown_root: Path | str, scenario_dir: Path | str) -> None:
        self.catalog = ScenarioCatalog(showdown_root=showdown_root)
        self.repository = ScenarioRepository(scenario_dir)

    def catalog_payload(self) -> dict[str, Any]:
        return self.catalog.payload()

    def generate_team(self, *, seed: int) -> dict[str, Any]:
        config = LocalShowdownConfig(showdown_root=self.catalog.showdown_root)
        with LocalShowdownEnv(config) as env:
            team = env.generate_scenario_team(seed=seed)
        side = self.catalog.side_from_generated_team(seed, team)
        return side.to_payload()

    def validate_payload(self, payload: Any) -> dict[str, Any]:
        scenario = validate_scenario(EndgameScenario.from_payload(payload), self.catalog)
        materialized = self._materialize(scenario, env_config=LocalShowdownConfig(showdown_root=self.catalog.showdown_root))
        return {"scenario": scenario.to_payload(), **materialized}

    def load(self, slug: str) -> dict[str, Any]:
        scenario = validate_scenario(self.repository.load(slug), self.catalog)
        return {"slug": slug, "scenario": scenario.to_payload()}

    def list(self) -> list[dict[str, Any]]:
        return self.repository.list()

    def save(self, slug: str, payload: Any) -> dict[str, Any]:
        validated = self.validate_payload(payload)
        scenario = EndgameScenario.from_payload(validated["scenario"])
        self.repository.save(slug, scenario)
        return {"slug": slug, **validated}

    def evaluate_root(
        self,
        payload: Any,
        *,
        checkpoint_path: Path | str,
        device: str | None = "cpu",
    ) -> dict[str, Any]:
        """Score the root action candidates with the checkpoint's latched encode contract."""

        scenario = validate_scenario(EndgameScenario.from_payload(payload), self.catalog)
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        model, result = load_transformer_checkpoint(checkpoint, map_location=device)
        env_config = env_config_with_checkpoint_masks(
            LocalShowdownConfig(showdown_root=self.catalog.showdown_root),
            feature_masks_from_model_config(result.model_config),
            context="scenario root evaluation",
            required_specs=observation_spec_from_model_config(result.model_config),
        )
        with LocalShowdownEnv(env_config) as env:
            env.reset_with_start_override(seed=scenario.seed, start_override=scenario_start_override(scenario))
            event = env.materialize_scenario_state(scenario_state=scenario_bridge_patch(scenario))
            # A Custom Game boundary asks both players for a choice, but a scenario root has one
            # explicitly designated actor. Keep the viewing perspective as provenance while the
            # model always scores the side the author selected to move.
            actor = scenario.side_to_move
            observation = env.observe(actor)
            state = env._state_for_player(actor)
            probabilities = evaluate_transformer_action_priors(
                model=model,
                result=result,
                observations=(observation,),
                device=device,
            )
            actions = [
                {
                    "index": index,
                    "label": choice_label(state, index),
                    "probability": probability,
                    "legal": bool(observation.legal_action_mask[index]),
                }
                for index, probability in enumerate(probabilities)
                if observation.legal_action_mask[index]
            ]
        actions.sort(key=lambda action: (-float(action["probability"]), int(action["index"])))
        for rank, action in enumerate(actions, start=1):
            action["rank"] = rank
            action["expected"] = action["label"] in scenario.objective.expected_root_actions
        return {
            "scenario": scenario.to_payload(),
            "materialization": _materialization_summary(event),
            "checkpoint": {
                "path": str(checkpoint),
                "policy_id": result.model_config.policy_id,
                "observation_schema_version": result.model_config.observation_schema_version,
                "transition_token_budget": result.model_config.transition_token_budget,
            },
            "synthetic_history": True,
            "perspective": scenario.perspective,
            "side_to_move": scenario.side_to_move,
            "actions": actions,
        }

    def _materialize(self, scenario: EndgameScenario, *, env_config: LocalShowdownConfig) -> dict[str, Any]:
        with LocalShowdownEnv(env_config) as env:
            env.reset_with_start_override(seed=scenario.seed, start_override=scenario_start_override(scenario))
            event = env.materialize_scenario_state(scenario_state=scenario_bridge_patch(scenario))
            action_summaries: dict[str, list[dict[str, Any]]] = {}
            for player in ("p1", "p2"):
                state = env._state_for_player(player)
                action_summaries[player] = [
                    {"index": index, "label": choice_label(state, index)}
                    for index, legal in enumerate(state.legal_action_mask)
                    if legal
                ]
        return {
            "validation": {
                "set_valid": True,
                "state_consistent": True,
                "replay_proven": scenario.replay_proven,
                "synthetic_history": True,
            },
            "materialization": _materialization_summary(event),
            "legal_actions": action_summaries,
        }


def _materialization_summary(event: Mapping[str, Any]) -> dict[str, Any]:
    state = event.get("state")
    requests = event.get("boundaryRequests")
    if not isinstance(state, Mapping) or not isinstance(requests, Mapping):
        raise ValueError("Scenario bridge returned malformed materialization event.")
    return {
        "turn": state.get("turn"),
        "sides": state.get("sides"),
        "requested_players": event.get("requested"),
    }
