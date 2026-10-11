"""One excluded deterministic searched-line correspondence decision unit.

This is NOT the scientific fidelity checker. It replays actual native ancestry
on one owned Showdown fixture, without regenerating native branches, repairing
HP/PP, tolerances, or pretending identical RNG integers couple two engines.
The fixed-damage/no-effect moves avoid that coupling problem for this fixture.
Full native PP correspondence is checked separately from public HP/status.
Private root/path states stay in this in-process diagnostic controller.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import time

from pokezero.dex import load_showdown_dex_cached, normalize_id
from pokezero.engine_search import EngineMctsPolicy
from pokezero.engine_world import world_battle_spec
from pokezero.env import BattleStartOverride
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.observation import OBSERVATION_SCHEMA_VERSION_V3, ObservationFeatureMasks
from pokezero.poke_engine_adapter import build_poke_engine_state
from pokezero.showdown import observation_spec_for_schema
from pokezero.showdown_fixture import FixturePokemon, pack_team
from pokezero.transitions_fold import FoldState

FIXTURE_SEED = 2026101101
FIXTURE_TEAMS = {
    "p1": (FixturePokemon("Primeape", ("Seismic Toss",), ability="Vital Spirit"),),
    "p2": (FixturePokemon("Blissey", ("Splash",), ability="Natural Cure"),),
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class DeterministicSearchedLineFixture:
    """Owns one explicit excluded fixture, never a scientific/source game."""

    def __init__(self, *, showdown_root):
        self.config = LocalShowdownConfig(showdown_root=showdown_root,
            observation_spec=observation_spec_for_schema(OBSERVATION_SCHEMA_VERSION_V3),
            feature_masks=ObservationFeatureMasks(transition_token_budget=32), set_belief_source=False)
        self.env = LocalShowdownEnv(self.config)
        self.override = BattleStartOverride(player_teams={slot: pack_team(team)
            for slot, team in FIXTURE_TEAMS.items()})
        self.checked = []
        try:
            self.env.reset_with_start_override(seed=FIXTURE_SEED, start_override=self.override)
            self.dex = load_showdown_dex_cached(self.config.resolved_showdown_root())
            materialization = self.env.public_materialization_state("p1")
            world = world_battle_spec(materialization, self.override, dex=self.dex)
            state = build_poke_engine_state(world.spec)
            observation = self.env.observe("p1")
            context = SimpleNamespace(public_materialization_state=materialization,
                observation=observation, player_id="p1", seed=FIXTURE_SEED,
                format_id="gen3customgame", battle_id="excluded-deterministic-searched-line")
            fold = FoldState.initial(perspective_slot="p1")
            fold.advance_in_place([event.raw_line for event in materialization.replay.public_events])
            self.position = dict(state_str=state.to_string(),
                row_inputs=EngineMctsPolicy._root_inputs_json(None, context),
                ctx=json.dumps({"p1": list(world.party_species["p1"]),
                    "p2": list(world.party_species["p2"]),
                    "turn": int(materialization.replay.turn_number)}),
                fold_state=fold.to_payload(), actives=(0, 0))
            self.root_snapshot = self.env.snapshot()
        except BaseException:
            self.env.close()
            raise

    def close(self):
        self.env.close()

    def _components(self, serialized):
        """Exact request-known fields, not a full-state equivalence claim."""
        import poke_engine
        state = poke_engine.State.from_string(serialized)
        differences = set()
        for slot, side in (("p1", state.side_one), ("p2", state.side_two)):
            request = self.env.public_materialization_state(slot).self_request
            party = request["side"]["pokemon"]
            active = next(mon for mon in party if mon["active"])
            native = side.pokemon[int(str(side.active_index))]
            condition = active["condition"].split()
            hp = int(condition[0].split("/")[0])
            expected_status = {"": "NONE", "par": "PARALYZE", "brn": "BURN",
                "psn": "POISON", "tox": "TOXIC", "slp": "SLEEP", "frz": "FREEZE"}.get(
                    condition[1] if len(condition) > 1 else "", "UNSUPPORTED")
            if native.hp != hp: differences.add("hp")
            if str(native.status).upper() != expected_status: differences.add("status")
            # Every live move slot in this fixture is visible in its owner's
            # actual request; no guessed opponent PP or repaired native PP.
            slots = request["active"][0]["moves"]
            for index, move in enumerate(slots):
                native_move = native.moves[index]
                if normalize_id(str(native_move.id)) != normalize_id(move["id"]):
                    differences.add("request")
                if native_move.pp != move["pp"]: differences.add("native_pp")
            if len(party) != 1 or str(state.weather).upper() != "NONE":
                differences.add("weather" if str(state.weather).upper() != "NONE" else "request")
        return differences

    def __call__(self, private_json):
        private = json.loads(private_json)
        if private["private_controller_only"] is not True:
            raise ValueError("private controller marker required")
        self.env.restore(deepcopy(self.root_snapshot))
        steps = private["steps"]
        differences = set()
        checked = 0
        expected_before = self.position["state_str"]
        if private["root_native_state"] != expected_before:
            differences.add("root")
        differences.update(self._components(expected_before))
        # Keep the whole actual ancestry denominator, including PP mismatch.
        # A public projection match is not allowed to erase that mismatch.
        for step in steps:
            if step["before"] != expected_before:
                differences.add("prestate")
                break
            actions = {}
            for slot, field, move in (("p1", "side_one_choice", "seismictoss"),
                    ("p2", "side_two_choice", "splash")):
                choice = step[field]
                if choice != {"kind": "move", "engine_id": move}:
                    differences.add("unsupported_action")
                    break
                candidates = self.env.observe(slot).metadata["action_candidates"]
                matches = [row["action_index"] for row in candidates
                    if row.get("kind") == "move" and normalize_id(str(row.get("move_id"))) == move
                    and self.env.legal_actions(slot)[row["action_index"]]]
                if len(matches) != 1:
                    differences.add("request")
                    break
                actions[slot] = matches[0]
            if len(actions) != 2 or set(self.env.requested_players()) != {"p1", "p2"}:
                differences.add("request")
                break
            self.env.step(actions)
            checked += 1
            differences.update(self._components(step["after"]))
            expected_before = step["after"]
        if expected_before != private["native_frontier"]:
            differences.add("engine_serialization")
        result = dict(status="MISMATCH" if differences else "MATCHED_PROJECTION_NOT_FULL_STATE",
            checked_transitions=checked, mismatch_components=sorted(differences))
        # Sanitized controller-local readout, not the private payload.
        self.checked.append(dict(evaluation_ordinal=private["evaluation_ordinal"],
            endpoint_sha256=sha(private["native_frontier"]),
            model_signed_value=private["model_signed_value"], **result))
        return json.dumps(result)


def validate_fixture_evidence(evidence):
    if not evidence["leaves"] or not any(row["path_boundaries"] >= 2 for row in evidence["leaves"]):
        raise ValueError("searched-line fixture did not capture an actual multi-boundary ancestor path")
    if any(row["result"]["checked_transitions"] != row["path_boundaries"] for row in evidence["leaves"]):
        raise ValueError("searched-line fixture did not replay its complete sampled ancestry")


def run_deterministic_fixture(*, native_model, tables_json, showdown_root):
    """One actual searched-line run; caller supplies an excluded fixture model.

    Returned results bind the raw native report, the sampled successful model
    events and their ACTUAL ancestor paths. No scientific checkpoint is loaded
    here. An operational error propagates; this function never retries.
    """
    import pokezero_search
    fixture = DeterministicSearchedLineFixture(showdown_root=showdown_root)
    handle = None
    try:
        position = fixture.position
        ctx = json.loads(position["ctx"])
        def unused_policy(raw):
            raise ValueError("searched-line fixture must not enter a continuation policy callback")
        handle = pokezero_search.NativeVisitedValueBank(sample_seed=17, worker=0,
            root_information_key="aa", original_deadline_at=time.perf_counter()+60,
            leaves_per_native_invocation=8, capture_searched_lines=True)
        started = time.perf_counter()
        raw_report = native_model.search_batched_multi_encoded(position["state_str"], 8, 4,
            tables_json, position["row_inputs"], position["ctx"],
            pokezero_search.FoldState.from_payload(position["fold_state"]),
            max_depth=2, seed=5, model_priors=True, use_opponent_priors=False, arm_priors=True,
            rollout_leaf_mode=None, rollout_policy="raw_argmax", rollouts=1, rollout_threads=1,
            rollout_max_plies=1, rollout_seed=101, record_joint_actions=True,
            raw_policy_callbacks=[unused_policy, unused_policy],
            raw_policy_request_orders=[ctx["p1"], ctx["p2"]], visited_value_bank=handle)
        report = json.loads(raw_report)
        evidence = json.loads(handle.check_searched_lines(fixture))
        if sha(raw_report) != evidence["selection_report_sha256"]:
            raise ValueError("actual native report binding changed")
        validate_fixture_evidence(evidence)
        return dict(schema="pokezero.search-over-raw.deterministic-searched-line-decision.v1",
            fixture_seed=FIXTURE_SEED, fixture_kind="excluded_customgame_fixed_damage_no_effect",
            fixture_model="random_weights_real_v3_shape_not_scientific_checkpoint",
            fixture_vocabulary_limitation="Splash uses catalog OOV safety-net tokens; not champion/model-coverage qualification",
            fixture_choices={"p1": "seismictoss", "p2": "splash"},
            elapsed_search_and_replay_seconds=time.perf_counter()-started,
            actual_search=dict(iterations=report["iterations"], model_evals=report["model_evals"],
                decision_nodes=report["decision_nodes"], chance_nodes=report["chance_nodes"]),
            evidence=evidence, controller_readout=fixture.checked,
            deterministic_scope_only=True, representative_runtime_evidence=False,
            full_native_state_correspondence=False, calibration_admission=False,
            historical_attempt_reused=False, no_retry=True,
            checked_projection_components=["active_hp", "active_status", "move_ids", "weather", "native_pp"],
            interpretation="Use the recorded mismatch components, not a general fidelity verdict. A native PP discrepancy may reflect the existing PP-ledger abstraction; full correspondence requires a separately reviewed effective-state/ledger contract. Random damage coupling is not tested.")
    finally:
        if handle is not None: handle.close()
        fixture.close()
