#!/usr/bin/env python3
"""Serve one source-bound MCTS policy over a local binary pipe.

This bootstrap intentionally lives outside the source root selected at launch.
It imports PokeZero only after putting that selected root first on sys.path, so
the native extension and Python search code belong to the declared policy
source rather than to the host rollout process.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
import random
import struct
import subprocess
import sys
from typing import Any, BinaryIO, Mapping


PROTOCOL_VERSION = "pokezero.isolated-mcts-policy.v1"
MAX_FRAME_BYTES = 64 * 1024 * 1024
STATS_FIELDS = (
    "decisions",
    "searched_decisions",
    "fallback_decisions",
    "model_evals",
    "total_iterations",
    "worlds_constructed",
    "worlds_searched",
    "prior_fallbacks",
    "decision_wall_seconds",
)


class WorkerError(RuntimeError):
    """The worker cannot provide a source-bound policy decision."""


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("host closed isolated policy pipe.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: BinaryIO) -> Mapping[str, Any]:
    size = struct.unpack(">Q", _read_exact(stream, 8))[0]
    if not size or size > MAX_FRAME_BYTES:
        raise WorkerError(f"invalid host frame size {size}.")
    try:
        payload = pickle.loads(_read_exact(stream, size))
    except (pickle.PickleError, EOFError) as error:
        raise WorkerError(f"cannot decode host frame: {error}") from error
    if not isinstance(payload, Mapping):
        raise WorkerError("host frame is not a mapping.")
    return payload


def write_frame(stream: BinaryIO, payload: Mapping[str, Any]) -> None:
    encoded = pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
    if not encoded or len(encoded) > MAX_FRAME_BYTES:
        raise WorkerError(f"invalid worker frame size {len(encoded)}.")
    stream.write(struct.pack(">Q", len(encoded)))
    stream.write(encoded)
    stream.flush()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_source_files(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def source_receipt(root: Path) -> dict[str, str]:
    """Require one clean Git checkout and reproduce the host tree hash recipe."""

    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError) as error:
        raise WorkerError(f"cannot verify isolated policy source root: {error}") from error
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise WorkerError("isolated policy source HEAD is not a full lowercase Git commit.")
    if dirty:
        raise WorkerError("isolated policy source tree is dirty.")
    paths = [
        root / value.decode("utf-8")
        for value in tracked
        if value and (root / value.decode("utf-8")).is_file()
    ]
    return {
        "commit": commit,
        "tree_sha256": _hash_source_files(root, paths),
        "tree_status": "clean_tracked_checkout",
    }


def _showdown_dependency_paths(root: Path) -> list[Path]:
    required = (
        root / "dist" / "sim" / "index.js",
        root / "dist" / "sim" / "dex.js",
        root / "dist" / "sim" / "dex-data.js",
        root / "dist" / "data" / "moves.js",
        root / "dist" / "data" / "pokedex.js",
        root / "dist" / "data" / "typechart.js",
        root / "dist" / "data" / "abilities.js",
        root / "dist" / "data" / "items.js",
        root / "dist" / "data" / "mods" / "gen3" / "moves.js",
        root / "dist" / "data" / "mods" / "gen3" / "scripts.js",
        root / "dist" / "data" / "mods" / "gen3" / "abilities.js",
        root / "dist" / "data" / "mods" / "gen3" / "items.js",
        root / "data" / "random-battles" / "gen3" / "sets.json",
        root / "dist" / "data" / "random-battles" / "gen3" / "teams.js",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise WorkerError(
            f"cannot bind isolated policy Showdown input {missing[0].relative_to(root)}."
        )
    paths = set(required)
    paths.update((root / "dist").rglob("*.js"))
    paths.update((root / "dist").rglob("*.json"))
    return sorted(path for path in paths if path.is_file())


def showdown_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _showdown_dependency_paths(root):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


class SnapshotAnnotationSource:
    """Per-request copy of the host's public Tier-2 annotation overlay."""

    def __init__(self) -> None:
        self._active = False
        self._overlay: dict[int, tuple[Any, ...]] = {}

    def set_snapshot(self, payload: object) -> None:
        if not isinstance(payload, Mapping):
            raise WorkerError("annotation snapshot is not a mapping.")
        active = payload.get("active")
        overlay = payload.get("overlay")
        if not isinstance(active, bool) or not isinstance(overlay, Mapping):
            raise WorkerError("annotation snapshot has an invalid shape.")
        copied: dict[int, tuple[Any, ...]] = {}
        for index, values in overlay.items():
            if not isinstance(index, int) or index < 0 or not isinstance(values, (tuple, list)):
                raise WorkerError("annotation snapshot has an invalid entry.")
            copied[index] = tuple(values)
        if not active and copied:
            raise WorkerError("inactive annotation snapshot must have an empty overlay.")
        self._active = active
        self._overlay = copied

    def active(self) -> bool:
        return self._active

    def overlay_for(self, player_id: str) -> dict[int, tuple[Any, ...]]:
        del player_id
        return dict(self._overlay)

    def boundary_state(self, player_id: str) -> Any:
        del player_id
        raise WorkerError(
            "isolated policy cannot run fold_cross_check against a host environment."
        )


def _require_public_context(context: object) -> None:
    player_id = getattr(context, "player_id", None)
    observations = getattr(context, "requested_observations", None)
    masks = getattr(context, "requested_legal_action_masks", None)
    trajectory = getattr(context, "trajectory", None)
    if not isinstance(player_id, str) or not isinstance(observations, Mapping) or not isinstance(
        masks, Mapping
    ):
        raise WorkerError("decision has no valid policy context.")
    if set(observations) != {player_id} or set(masks) != {player_id}:
        raise WorkerError("host leaked another requested player's private boundary data.")
    for step in getattr(trajectory, "steps", ()):
        if getattr(step, "player_id", None) != player_id and getattr(
            step, "observation", None
        ) is not None:
            raise WorkerError("host leaked another player's historic observation.")


def _stats_payload(stats: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in STATS_FIELDS:
        value = getattr(stats, field_name, 0.0 if field_name == "decision_wall_seconds" else 0)
        payload[field_name] = float(value) if field_name == "decision_wall_seconds" else int(value)
    return payload


def _decision_payload(decision: Any) -> dict[str, Any]:
    metadata = getattr(decision, "metadata", {})
    if not isinstance(metadata, Mapping):
        raise WorkerError("engine policy returned decision metadata that is not a mapping.")
    return {
        "action_index": int(getattr(decision, "action_index")),
        "policy_id": str(getattr(decision, "policy_id")),
        "action_probability": getattr(decision, "action_probability", None),
        "value_estimate": getattr(decision, "value_estimate", None),
        "metadata": dict(metadata),
    }


def _worker_start(message: Mapping[str, Any]) -> tuple[Any, SnapshotAnnotationSource, dict[str, Any]]:
    if message.get("type") != "start" or message.get("protocol_version") != PROTOCOL_VERSION:
        raise WorkerError("host did not begin the expected isolated policy protocol.")
    policy = message.get("policy")
    config = message.get("worker_config")
    if not isinstance(policy, Mapping) or not isinstance(config, Mapping):
        raise WorkerError("isolated policy start message has an invalid shape.")
    source_root_value = config.get("source_root")
    showdown_root_value = config.get("showdown_root")
    bootstrap_sha256 = config.get("worker_bootstrap_sha256")
    if (
        not isinstance(source_root_value, str)
        or not isinstance(showdown_root_value, str)
        or not isinstance(bootstrap_sha256, str)
    ):
        raise WorkerError(
            "isolated policy worker config requires source_root, showdown_root, and "
            "worker_bootstrap_sha256."
        )
    if _sha256_file(Path(__file__).resolve()) != bootstrap_sha256:
        raise WorkerError("isolated policy worker bootstrap does not match its declared hash.")
    source_root = Path(source_root_value).expanduser().resolve()
    showdown_root = Path(showdown_root_value).expanduser().resolve()
    receipt = source_receipt(source_root)
    if receipt["commit"] != str(policy.get("source_commit", "")):
        raise WorkerError("isolated policy source commit does not match its declared policy.")
    if receipt["tree_sha256"] != str(policy.get("source_tree_sha256", "")):
        raise WorkerError("isolated policy source tree hash does not match its declared policy.")
    if showdown_sha256(showdown_root) != str(policy.get("showdown_source_sha256", "")):
        raise WorkerError("isolated policy Showdown source does not match its declared policy.")
    config_payload = policy.get("config")
    if not isinstance(config_payload, Mapping):
        raise WorkerError("isolated policy has no engine config mapping.")
    if config_payload.get("leaf_eval") != "model" or config_payload.get("strict_fallbacks") is not True:
        raise WorkerError("isolated policy must use strict model-leaf MCTS.")
    if config_payload.get("fold_cross_check") is True:
        raise WorkerError(
            "isolated policy refuses fold_cross_check because it has no host environment."
        )

    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(source_root / "scripts"))
    try:
        from engine_build_fingerprint import assert_fresh, compute_fingerprint
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy
        from pokezero.randbat import load_gen3_randbat_source_cached
    except Exception as error:
        raise WorkerError(f"cannot import declared isolated policy source: {error}") from error
    assert_fresh()
    fingerprint = str(compute_fingerprint()["fingerprint"])
    if fingerprint != str(policy.get("engine_fingerprint", "")):
        raise WorkerError("isolated policy engine fingerprint does not match its declared policy.")
    annotations = SnapshotAnnotationSource()
    try:
        engine_config = EngineMctsConfig(**dict(config_payload))
        engine_policy = EngineMctsPolicy(
            dex=load_showdown_dex_cached(showdown_root),
            set_source=load_gen3_randbat_source_cached(showdown_root),
            config=engine_config,
            policy_id=str(policy.get("policy_id", "")),
            annotation_source=annotations,
        )
    except Exception as error:
        raise WorkerError(f"cannot construct isolated MCTS policy: {error}") from error
    receipt.update(
        {
            "engine_fingerprint": fingerprint,
            "policy": dict(policy),
            "worker_bootstrap_sha256": bootstrap_sha256,
        }
    )
    return engine_policy, annotations, receipt


def _serve() -> int:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    try:
        start = read_frame(stdin)
        policy, annotations, receipt = _worker_start(start)
    except Exception as error:
        write_frame(stdout, {"type": "error", "message": str(error)})
        return 2
    write_frame(stdout, {"type": "hello", "receipt": receipt})
    while True:
        try:
            message = read_frame(stdin)
            kind = message.get("type")
            if kind == "close":
                write_frame(stdout, {"type": "close"})
                return 0
            if kind == "reset":
                policy.reset()
                write_frame(stdout, {"type": "reset"})
                continue
            if kind != "decide":
                raise WorkerError(f"unsupported isolated policy request {kind!r}.")
            context = message.get("context")
            _require_public_context(context)
            annotations.set_snapshot(message.get("annotation"))
            rng = random.Random()
            rng.setstate(message.get("rng_state"))
            decision = policy.select_action_with_context(context, rng=rng)
            write_frame(
                stdout,
                {
                    "type": "decision",
                    "decision": _decision_payload(decision),
                    "stats": _stats_payload(policy.stats),
                },
            )
        except Exception as error:
            write_frame(stdout, {"type": "error", "message": str(error)})


if __name__ == "__main__":
    raise SystemExit(_serve())
