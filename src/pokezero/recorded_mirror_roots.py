"""Frozen TRAIN-only mirror roots for separately admitted cost quarantine.

This module consumes recorded protocol and observations only. It never starts,
steps or restores a simulator, runs a model, computes outcomes, or replaces a
failed root. Native search timing and scientific panels need separate admission.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from .mirror_value_probe import CHAMPION_SHA, cohort, digest, file_sha

SELECTION_SALT = "a4-cost-quarantine-20261029-v1"
COLLECTION_SHA = "a71d93f84ef39d399cd2c6cf8ee2b303d9e7435e30f67c0c6c746f7c94501ae6"
OBSERVATION_FIELDS = ("categorical_ids", "numeric_features", "token_type_ids",
                      "attention_mask", "legal_action_mask")


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
    for row in boundaries:
        if row.get("event") == "end":
            continue
        if (row.get("schema") != "pokezero.mirror_boundary.v1"
                or row.get("battle_id") != battle["battle_id"] or row.get("seed") != battle["seed"]):
            raise ValueError("source boundary identity mismatch")
        if seat in row["requested_players"]:
            candidates.append(row)
    if not candidates:
        raise ValueError("selected training battle has no requested root; no replacements")
    return min(candidates, key=lambda row: digest([salt, "boundary", battle["battle_id"], seat, row["root_id"]]))


def freeze_cost_roots(collection: str | Path, *, expected_sha256: str = COLLECTION_SHA,
                      salt: str = SELECTION_SALT) -> dict[str, Any]:
    """Read only fixed source receipts/boundaries; do not open feature arrays."""
    root = Path(collection)
    if file_sha(root / "collection.json") != expected_sha256:
        raise ValueError("collection SHA mismatch")
    manifest = json.loads((root / "collection.json").read_text())
    if (manifest.get("schema") != "pokezero.mirror_collection.v1"
            or manifest.get("status") != "COMPLETE_FIXED_COHORT"
            or manifest["config"]["champion_sha256"] != CHAMPION_SHA):
        raise ValueError("completed champion mirror collection required")
    if file_sha(root / "cohort.json") != manifest["cohort_sha256"]:
        raise ValueError("cohort SHA mismatch")
    expected = cohort(manifest["config"]["root"], manifest["config"]["battles"])
    source = json.loads((root / "cohort.json").read_text())
    if source["battles"] != expected or source["roster_sha256"] != digest(expected):
        raise ValueError("original preassigned split/roster mismatch")
    rows = select_training_battles(expected, salt=salt)
    selected = []
    for battle in rows:
        rel = f"battle-{battle['index']:04d}/battle.json"
        receipt_path = root / rel
        receipt_sha = file_sha(receipt_path)
        if receipt_sha != manifest["battle_receipt_hashes"][rel]:
            raise ValueError("battle receipt SHA mismatch")
        receipt = json.loads(receipt_path.read_text())
        if any(receipt[name] != battle[name] for name in ("index", "battle_id", "seed", "split")):
            raise ValueError("battle split/identity mismatch")
        path = receipt_path.parent / "boundaries.jsonl.gz"
        boundary_sha = file_sha(path)
        if boundary_sha != receipt["files"][path.name]:
            raise ValueError("boundary artifact SHA mismatch")
        with gzip.open(path, "rt") as stream:
            boundaries = [json.loads(line) for line in stream]
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
    if row.get("schema") != "pokezero.mirror_boundary.v1" or row.get("event") is not None:
        raise ValueError("recorded nonterminal request boundary required")
    if row["protocol_start"] != len(env.protocol_lines):
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
    return actual


def reconstruct_cost_root(config: Any, collection: str | Path, root: dict[str, Any]) -> tuple[Any, Any]:
    """Reconstruct only the frozen root; failure is terminal, never a skip."""
    if root.get("split") != "train" or root.get("cost_seat") not in ("p1", "p2"):
        raise ValueError("fixed TRAIN cost root required")
    path = Path(collection) / root["boundaries_relative_path"]
    if file_sha(path) != root["boundaries_file_sha256"]:
        raise ValueError("frozen source boundary artifact changed")
    env = new_recorded_environment(config, seed=root["seed"])
    try:
        with gzip.open(path, "rt") as stream:
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
