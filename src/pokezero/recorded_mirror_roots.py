"""Frozen TRAIN-only mirror roots for separately admitted cost quarantine.

This module consumes recorded protocol and observations only. It never starts,
steps or restores a simulator, runs a model, computes outcomes, or replaces a
failed root. Native search timing and scientific panels need separate admission.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from .mirror_value_probe import CHAMPION_SHA, cohort, digest

SELECTION_SALT = "a4-cost-quarantine-20261029-v1"
COLLECTION_SHA = "a71d93f84ef39d399cd2c6cf8ee2b303d9e7435e30f67c0c6c746f7c94501ae6"
OBSERVATION_FIELDS = ("categorical_ids", "numeric_features", "token_type_ids",
                      "attention_mask", "legal_action_mask")


CHAMPION_CONTRACT = {'belief_set_source_hash': 'f5a5265143d423af', 'categorical_vocab_size': 916, 'category_vocab_sha256': '66627e048ac8a07053599b32398211579ec3216f51a390f74442ebb4c64650e8', 'checkpoint_sha256': '0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be', 'feature_masks': {'exact_state': True, 'feature_pack_last_move': True, 'investment_belief_narrowing': False, 'item_belief_narrowing': False, 'stats_block': True, 'tier2_investment': True, 'tier2_residuals': True, 'transition_token_budget': 0}, 'model_config_sha256': '35718643b3e8580f8a9ffb9dd48986b469d15f7ee5e27719ff39baae47ed0043', 'schema': 'pokezero.value_probe_checkpoint_contract.v1', 'token_type_vocab_size': 16, 'value_activation': 'tanh', 'value_calibration_transform': None, 'width': 512}


def recorded_requested_players(requests):
    """Match production request actionability without importing simulator code.

    Tests compare this pure metadata projection to the production helper.
    """
    if not isinstance(requests, dict) or set(requests) - {"p1", "p2"}:
        raise ValueError("only legal owning request seats required")
    result = []
    for seat in ("p1", "p2"):
        request = requests.get(seat)
        if request is None:
            continue
        if not isinstance(request, dict) or request.get("side", {}).get("id") != seat:
            raise ValueError("owning request identity mismatch")
        if request.get("wait") or request.get("teamPreview"):
            continue
        force_switch, active = request.get("forceSwitch"), request.get("active")
        if ((isinstance(force_switch, list) and any(bool(slot) for slot in force_switch))
                or (isinstance(active, list) and bool(active))):
            result.append(seat)
    return tuple(result)


def validate_boundary_roles(row, battle):
    """Validate even ineligible rows before any selection can exclude them."""
    if (row.get("schema") != "pokezero.mirror_boundary.v1"
            or row.get("battle_id") != battle["battle_id"] or row.get("seed") != battle["seed"]):
        raise ValueError("source boundary identity mismatch")
    requested = row.get("requested_players")
    if (not isinstance(requested, list) or not requested or len(set(requested)) != len(requested)
            or any(seat not in ("p1", "p2") for seat in requested)
            or tuple(requested) != recorded_requested_players(row["requests"])):
        raise ValueError("unique requested roles must match actionable source requests")
    if (set(row.get("observations", {})) != set(requested)
            or set(row.get("actions", {})) != set(requested)
            or type(row.get("paired")) is not bool or row["paired"] != (len(requested) == 2)):
        raise ValueError("exact requested observation/action role coverage required")
    for seat in requested:
        obs, action = row["observations"][seat], row["actions"][seat]
        for name, shape, kinds in (("categorical_ids", (23,41), "iu"),
                ("numeric_features", (23,132), "iuf"), ("token_type_ids", (23,), "iu"),
                ("attention_mask", (23,), "b"), ("legal_action_mask", (9,), "b")):
            value = np.asarray(obs[name])
            if value.shape != shape or value.dtype.kind not in kinds or not np.isfinite(value).all():
                raise ValueError("invalid recorded observation: " + seat + "." + name)
        if (type(action) is not int or not 0 <= action < 9
                or not obs["legal_action_mask"][action] or not any(obs["attention_mask"])):
            raise ValueError("committed source action must be legal")
    if (type(row.get("decision_round_index")) is not int or row["decision_round_index"] < 0
            or row["root_id"] != digest([row["battle_id"], row["decision_round_index"],
                                          row["public_prefix_sha256"], row["requests"]])):
        raise ValueError("source root digest/round mismatch")


def select_training_battles(roster: list[dict[str, Any]], *, count: int = 20,
                            salt: str = SELECTION_SALT) -> list[dict[str, Any]]:
    """Choose distinct training battles before reading their roots or values."""
    if type(count) is not int or count < 20 or count % 2 or not salt:
        raise ValueError("at least20 even fixed battle count and salt required")
    if len({row["battle_id"] for row in roster}) != len(roster):
        raise ValueError("unique source battles required")
    eligible = [row for row in roster if row["split"] == "train"]
    if len(eligible) < count:
        raise ValueError("insufficient TRAIN battles; no replacements")
    ranked = sorted(eligible, key=lambda row: digest([salt, "battle", row["battle_id"]]))[:count]
    return [dict(row, cost_ordinal=index, cost_seat="p1" if index % 2 == 0 else "p2")
            for index, row in enumerate(ranked)]


def select_recorded_boundary(boundaries: list[dict[str, Any]], battle: dict[str, Any],
                              *, salt: str = SELECTION_SALT) -> dict[str, Any]:
    """Fixed hash rank over this seat's recorded requests; never filter by cost."""
    seat = battle["cost_seat"]
    candidates = []
    seen_rounds, seen_roots, prefix = set(), set(), []
    ended = False
    for row in boundaries:
        if ended:
            raise ValueError("records after terminal boundary")
        if row.get("event") == "end":
            ended = True
            if row["protocol_start"] != len(prefix):
                raise ValueError("terminal source prefix continuity mismatch")
            prefix.extend(row["protocol_delta"])
            if row["protocol_end"] != len(prefix):
                raise ValueError("terminal source prefix length mismatch")
            continue
        validate_boundary_roles(row, battle)
        index, identity = row["decision_round_index"], row["root_id"]
        if index in seen_rounds or identity in seen_roots or index != len(seen_rounds):
            raise ValueError("unique chronological rounds and roots required")
        seen_rounds.add(index); seen_roots.add(identity)
        if row["protocol_start"] != len(prefix):
            raise ValueError("source prefix continuity mismatch")
        prefix.extend(row["protocol_delta"])
        if row["protocol_end"] != len(prefix) or digest(prefix) != row["public_prefix_sha256"]:
            raise ValueError("source prefix digest mismatch")
        if seat in row["requested_players"]:
            candidates.append(row)
    if not ended:
        raise ValueError("complete recorded battle with final boundary required")
    if not candidates:
        raise ValueError("selected training battle has no requested root; no replacements")
    return min(candidates, key=lambda row: digest([salt, "boundary", battle["battle_id"], seat, row["root_id"]]))


def freeze_cost_roots(collection: str | Path, *, expected_sha256: str = COLLECTION_SHA,
                      salt: str = SELECTION_SALT) -> dict[str, Any]:
    """Read only fixed source receipts/boundaries; do not open feature arrays."""
    root = Path(collection)
    manifest_blob = (root / "collection.json").read_bytes()
    if hashlib.sha256(manifest_blob).hexdigest() != expected_sha256:
        raise ValueError("collection SHA mismatch")
    manifest = json.loads(manifest_blob)
    if (manifest.get("schema") != "pokezero.mirror_collection.v1"
            or manifest.get("status") != "COMPLETE_FIXED_COHORT"
            or manifest["config"]["champion_sha256"] != CHAMPION_SHA):
        raise ValueError("completed champion mirror collection required")
    cohort_blob = (root / "cohort.json").read_bytes()
    if hashlib.sha256(cohort_blob).hexdigest() != manifest["cohort_sha256"]:
        raise ValueError("cohort SHA mismatch")
    expected = cohort(manifest["config"]["root"], manifest["config"]["battles"])
    source = json.loads(cohort_blob)
    if (source.get("schema") != "pokezero.mirror_cohort.v1"
            or source["battles"] != expected or source["roster_sha256"] != digest(expected)
            or source.get("config") != manifest["config"]
            or source.get("config_sha256") != digest(manifest["config"])
            or manifest.get("config_sha256") != digest(manifest["config"])
            or len(manifest.get("battles", [])) != len(expected)
            or any(any(actual.get(k) != value for k, value in wanted.items())
                   for wanted, actual in zip(expected, manifest["battles"]))):
        raise ValueError("original preassigned split/roster/config mismatch")
    rows = select_training_battles(expected, salt=salt)
    selected = []
    for battle in rows:
        rel = f"battle-{battle['index']:04d}/battle.json"
        receipt_path = root / rel
        receipt_blob = receipt_path.read_bytes()
        receipt_sha = hashlib.sha256(receipt_blob).hexdigest()
        if receipt_sha != manifest["battle_receipt_hashes"][rel]:
            raise ValueError("battle receipt SHA mismatch")
        receipt = json.loads(receipt_blob)
        if (receipt.get("schema") != "pokezero.mirror_battle.v1"
                or any(receipt.get(name) != battle[name] for name in ("index", "battle_id", "seed", "split", "split_key"))
                or receipt.get("contract") != CHAMPION_CONTRACT):
            raise ValueError("battle split/identity/champion contract mismatch")
        path = receipt_path.parent / "boundaries.jsonl.gz"
        boundary_blob = path.read_bytes()
        boundary_sha = hashlib.sha256(boundary_blob).hexdigest()
        if boundary_sha != receipt["files"][path.name]:
            raise ValueError("boundary artifact SHA mismatch")
        boundaries = [json.loads(line) for line in gzip.decompress(boundary_blob).splitlines()]
        boundary = select_recorded_boundary(boundaries, battle, salt=salt)
        selected.append({name: battle[name] for name in
            ("index", "battle_id", "seed", "split", "cost_ordinal", "cost_seat")})
        selected[-1].update(decision_round_index=boundary["decision_round_index"],
            root_id=boundary["root_id"], boundary_sha256=digest(boundary),
            observation_sha256=digest(boundary["observations"][battle["cost_seat"]]),
            protocol_prefix_sha256=boundary["public_prefix_sha256"],
            requests_sha256=digest(boundary["requests"]), battle_receipt_sha256=receipt_sha,
            boundaries_file_sha256=boundary_sha, boundaries_relative_path=str(path.relative_to(root)))
    return dict(schema="pokezero.A4-saved-cost-roots.v1", collection_sha256=expected_sha256,
        cohort_sha256=manifest["cohort_sha256"], selection_salt=salt, roots=selected,
        selected_root_sha256=digest(selected), count=len(selected),
        rule="20 distinct TRAIN battles by hash, alternating10p1/10p2; one requested boundary per battle by hash; no cost/value/outcome filtering or replacements",
        scope="cost quarantine only; no scientific games, outcome metrics, model forward, search or sibling evidence",
        new_games=0, simulator_steps=0, native_search_calls=0)


def new_recorded_environment(config: Any, *, seed: int) -> Any:
    """Initialize production parser/belief state without any simulator process."""
    from .local_showdown import LocalShowdownEnv
    from .showdown import _ReplayParser
    env = LocalShowdownEnv(config)
    env._battle_id = f"local-gen3randombattle-{seed}"
    env._parser = _ReplayParser(env._battle_id, complete_prefix=True,
                                hp_visibility={"p1": "exact", "p2": "exact"})
    if env._process is not None or env._lines:
        raise ValueError("fresh offline environment required")
    return env


def feed_recorded_boundary(env: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Advance parser/request history and prove every recorded actor observation.

    Mixed trusted source lines are routed through the same production event
    ingestion as the collector. They are NOT a public wire. The production
    public_materialization_state seam strips requests before search receives it.
    """
    if env._process is not None or env._battle_token is not None:
        raise ValueError("offline reconstruction forbids a simulator process/token")
    if row.get("event") is not None:
        raise ValueError("recorded nonterminal request boundary required")
    validate_boundary_roles(row, {"battle_id":row.get("battle_id"), "seed":row.get("seed")})
    if (row["protocol_start"] != len(env.protocol_lines)
            or row["decision_round_index"] != getattr(env, "_recorded_cost_next_round", 0)):
        raise ValueError("source protocol continuity mismatch")
    for line in row["protocol_delta"]:
        if not isinstance(line, str) or not line:
            raise ValueError("nonempty source protocol lines required")
        stream = "omniscient"
        if line.startswith("|request|"):
            request = json.loads(line[len("|request|"):])
            stream = request.get("side", {}).get("id")
            if stream not in ("p1", "p2"):
                raise ValueError("private request has no owning seat")
        env._apply_event(dict(type="stream", stream=stream, lines=[line]))
    if (row["protocol_end"] != len(env.protocol_lines)
            or digest(env.protocol_lines) != row["public_prefix_sha256"]
            or env._latest_requests != row["requests"]
            or tuple(env.requested_players()) != tuple(row["requested_players"])):
        raise ValueError("recorded source prefix/request boundary mismatch")
    expected_root = digest([row["battle_id"], row["decision_round_index"],
                            row["public_prefix_sha256"], row["requests"]])
    if row["root_id"] != expected_root:
        raise ValueError("source root SHA mismatch")
    actual = {}
    for seat in row["requested_players"]:
        observation = env.observe(seat)
        reference = row["observations"][seat]
        for name in OBSERVATION_FIELDS:
            if not np.array_equal(np.asarray(getattr(observation, name)), np.asarray(reference[name])):
                raise ValueError("recorded Python observation differs: " + seat + "." + name)
        if observation.metadata.get("turn_number") != reference["turn_number"]:
            raise ValueError("recorded Python turn differs")
        actual[seat] = observation
    env._recorded_cost_next_round = row["decision_round_index"] + 1
    return actual


def reconstruct_cost_root(config: Any, collection: str | Path, root: dict[str, Any]) -> tuple[Any, Any]:
    """Reconstruct only the frozen root; failure is terminal, never a skip."""
    if root.get("split") != "train" or root.get("cost_seat") not in ("p1", "p2"):
        raise ValueError("fixed TRAIN cost root required")
    collection_root = Path(collection).resolve(strict=True)
    path = (collection_root / root["boundaries_relative_path"]).resolve(strict=True)
    if not path.is_relative_to(collection_root):
        raise ValueError("source boundary must remain inside collection")
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != root["boundaries_file_sha256"]:
        raise ValueError("frozen source boundary artifact changed")
    env = new_recorded_environment(config, seed=root["seed"])
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(blob), mode="rb") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("event") == "end":
                    raise ValueError("selected root absent before terminal")
                if row["battle_id"] != root["battle_id"] or row["seed"] != root["seed"]:
                    raise ValueError("source battle identity changed")
                observations = feed_recorded_boundary(env, row)
                if row["decision_round_index"] == root["decision_round_index"]:
                    if (digest(row) != root["boundary_sha256"] or row["root_id"] != root["root_id"]
                            or digest(row["requests"]) != root["requests_sha256"]
                            or digest(row["observations"][root["cost_seat"]]) != root["observation_sha256"]):
                        raise ValueError("frozen root or observation changed")
                    return env, observations[root["cost_seat"]]
        raise ValueError("selected root missing; no replacements")
    except BaseException:
        env.close()
        raise
