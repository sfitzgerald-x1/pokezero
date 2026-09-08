#!/usr/bin/env python3
"""Run a frozen, fresh, mirrored MCTS-versus-MCTS comparison.

The manifest is intentionally small but complete enough to be an experiment
contract: it names two explicit model-leaf MCTS configurations, one checkpoint
digest, source/build identities, a seed list, and the decision cap.  A
source-different incumbent is allowed only through the explicit isolated-build
mode, where it serves decisions from its declared source-bound process.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero.mcts_eval.head_to_head import (  # noqa: E402
    HeadToHeadError,
    MctsPolicySpec,
    PublicOnlyMctsPolicy,
    complete_pair,
    load_pair,
    play_mirrored_pair,
    summarize_complete_pairs,
    write_game_immutable,
)


MANIFEST_SCHEMA_VERSION = "pokezero.mcts-h2h-manifest.v1"
COMPLETE_SCHEMA_VERSION = "pokezero.mcts-h2h-complete.v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_source_files(repo_root: Path, paths: list[Path]) -> str:
    """Hash source files with their repository-relative names and bytes."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(repo_root).as_posix()):
        relative = path.relative_to(repo_root).as_posix()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _source_provenance() -> dict[str, str]:
    """Bind Python/runtime source to a clean checkout or an explicit content receipt."""

    stamped = os.environ.get("POKEZERO_COMMIT", "").strip().lower()
    git_metadata = REPO_ROOT / ".git"
    if git_metadata.exists():
        try:
            head = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().lower()
            dirty = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            tracked = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
                check=True,
                capture_output=True,
            ).stdout.split(b"\0")
        except (OSError, subprocess.CalledProcessError) as error:
            raise HeadToHeadError(
                f"cannot verify public source checkout before a scored comparison: {error}"
            ) from error
        if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
            raise HeadToHeadError("public source checkout HEAD is not a full lowercase Git commit id.")
        if stamped and stamped != head:
            raise HeadToHeadError(
                f"POKEZERO_COMMIT={stamped} does not match the executing checkout {head}."
            )
        if dirty:
            changed = ", ".join(line[3:].strip() for line in dirty.splitlines()[:8])
            raise HeadToHeadError(
                "refusing to score a dirty tracked public source tree"
                f" ({changed or 'tracked changes detected'})."
            )
        paths = [
            REPO_ROOT / value.decode("utf-8")
            for value in tracked
            if value and (REPO_ROOT / value.decode("utf-8")).is_file()
        ]
        return {
            "commit": head,
            "tree_sha256": _hash_source_files(REPO_ROOT, paths),
            "tree_status": "clean_tracked_checkout",
        }

    if len(stamped) != 40 or any(character not in "0123456789abcdef" for character in stamped):
        raise HeadToHeadError(
            "an image without .git must set POKEZERO_COMMIT to its full lowercase source commit."
        )
    roots = (REPO_ROOT / "src" / "pokezero", REPO_ROOT / "scripts")
    paths = [
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    # LocalShowdownEnv launches scripts/battle_bridge.mjs, which imports the
    # sibling request-boundary module. Include every in-repository JS module so
    # the image receipt covers that executable battle seam as well as Python.
    paths.extend(
        path
        for path in (REPO_ROOT / "scripts").glob("*.mjs")
        if path.is_file()
    )
    pyproject = REPO_ROOT / "pyproject.toml"
    if pyproject.is_file():
        paths.append(pyproject)
    if not paths:
        raise HeadToHeadError("cannot hash executable source in image without .git.")
    return {
        "commit": stamped,
        "tree_sha256": _hash_source_files(REPO_ROOT, paths),
        "tree_status": "explicit_hash_without_git_python_and_bridge",
    }


def _showdown_dependency_paths(root: Path) -> list[Path]:
    """Return every built Showdown byte the Gen 3 battle oracle can load."""

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
        raise HeadToHeadError(
            "cannot bind Showdown source; required runtime input is missing: "
            f"{missing[0].relative_to(root)}."
        )
    paths = set(required)
    # Dex.forGen(3) can resolve inherited mod layers through the built simulator.
    # Hash every built source byte, not a brittle hand-maintained transitive-load list.
    paths.update((root / "dist").rglob("*.js"))
    paths.update((root / "dist").rglob("*.json"))
    return sorted(path for path in paths if path.is_file())


def _showdown_source_provenance(showdown_root: str | Path) -> dict[str, Any]:
    """Require a clean checkout and bind all runtime-relevant Showdown content."""

    root = Path(showdown_root).expanduser().resolve()
    try:
        top_level = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise HeadToHeadError(
            f"cannot bind Showdown source identity from {root}: {error}"
        ) from error
    if top_level != root:
        raise HeadToHeadError("--showdown-root must be the root of its Showdown Git checkout.")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise HeadToHeadError("Showdown HEAD is not a full lowercase Git commit id.")
    if dirty:
        raise HeadToHeadError(
            "Showdown checkout is dirty; commit or restore it before an MCTS-vs-MCTS result."
        )
    digest = hashlib.sha256()
    for path in _showdown_dependency_paths(root):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return {"content_sha256": digest.hexdigest(), "git_commit": commit, "git_clean": True}


def _declared_showdown_source_sha256(manifest: Mapping[str, Any]) -> str:
    value = str(manifest.get("showdown_source_sha256", ""))
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise HeadToHeadError(
            "manifest.showdown_source_sha256 must be the 64-character hash of the "
            "clean Showdown runtime inputs."
        )
    return value


def _declared_source_tree_sha256(manifest: Mapping[str, Any]) -> str:
    value = str(manifest.get("source_tree_sha256", ""))
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise HeadToHeadError(
            "manifest.source_tree_sha256 must be the 64-character hash of the executing "
            "PokeZero source/image receipt."
        )
    return value


def _durable_output_root(value: str | Path) -> Path:
    """Keep resumable evidence outside the source tree that must remain clean."""

    root = Path(value).expanduser().resolve()
    try:
        root.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return root
    raise HeadToHeadError(
        "--out-dir must be outside the PokeZero source checkout; write durable games to "
        "/shared (or another external evidence root) so a resumed result never dirties its "
        "own source identity."
    )


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create ``path`` or require byte-identical prior contents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    expected = _canonical_bytes(payload)
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing != expected:
            raise HeadToHeadError(
                f"refusing to replace extant immutable artifact {path}; its contents differ."
            )
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != expected:
                raise HeadToHeadError(
                    f"refusing to replace concurrently-created artifact {path}; it differs."
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HeadToHeadError(f"{label} must be a JSON object.")
    return value


def _load_manifest(path: str | Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HeadToHeadError(f"cannot read MCTS-vs-MCTS manifest: {error}") from error
    manifest = _mapping(payload, label="manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise HeadToHeadError(
            f"manifest schema {manifest.get('schema_version')!r} is not "
            f"{MANIFEST_SCHEMA_VERSION!r}."
        )
    return manifest


def _runtime_spec(
    raw: Mapping[str, Any],
    *,
    role: str,
    checkpoint: str,
    checkpoint_sha256: str,
    source_commit: str,
    source_tree_sha256: str,
    engine_fingerprint: str,
    showdown_source_sha256: str,
    model_path: str,
    tables_path: str,
    device: str,
) -> tuple[MctsPolicySpec, Any]:
    """Build and freeze one model-leaf policy configuration from the manifest."""

    from pokezero.engine_search import EngineMctsConfig

    for field, actual in (
        ("source_commit", source_commit),
        ("engine_fingerprint", engine_fingerprint),
        ("checkpoint_sha256", checkpoint_sha256),
    ):
        declared = str(raw.get(field, ""))
        if declared != actual:
            raise HeadToHeadError(
                f"{role}.{field}={declared!r} does not match the active verified value {actual!r}."
            )
    settings = dict(_mapping(raw.get("engine_config"), label=f"{role}.engine_config"))
    if settings.pop("leaf_eval", None) != "model":
        raise HeadToHeadError(f"{role} must explicitly declare engine_config.leaf_eval='model'.")
    if settings.pop("strict_fallbacks", None) is not True:
        raise HeadToHeadError(
            f"{role} must explicitly declare engine_config.strict_fallbacks=true."
        )
    forbidden = {"checkpoint_path", "model_path", "tables_path", "model_device"}.intersection(settings)
    if forbidden:
        raise HeadToHeadError(
            f"{role}.engine_config may not override source-bound artifacts: {sorted(forbidden)}."
        )
    try:
        config = EngineMctsConfig(
            leaf_eval="model",
            strict_fallbacks=True,
            checkpoint_path=checkpoint,
            model_path=model_path,
            tables_path=tables_path,
            model_device=device,
            **settings,
        )
    except (TypeError, ValueError) as error:
        raise HeadToHeadError(f"invalid {role}.engine_config: {error}") from error
    frozen = asdict(config)
    try:
        json.dumps(frozen, sort_keys=True)
    except TypeError as error:
        raise HeadToHeadError(f"{role} configuration is not serializable: {error}") from error
    spec = MctsPolicySpec(
        config_id=str(raw.get("config_id", "")),
        policy_id=str(raw.get("policy_id", "")),
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        engine_fingerprint=engine_fingerprint,
        checkpoint_sha256=checkpoint_sha256,
        showdown_source_sha256=showdown_source_sha256,
        config=frozen,
    )
    return spec, config


def _seeds(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    values = manifest.get("seeds")
    if not isinstance(values, list) or not values:
        raise HeadToHeadError("manifest.seeds must be a non-empty JSON list.")
    seeds = tuple(int(value) for value in values)
    if len(set(seeds)) != len(seeds):
        raise HeadToHeadError("manifest.seeds contains a duplicate; each mirrored pair needs one key.")
    return seeds


def _required_sha256(value: object, *, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise HeadToHeadError(f"{label} must be a 64-character lowercase SHA-256 hash.")
    return digest


def _isolated_incumbent_identity(raw: Mapping[str, Any]) -> tuple[str, str, str]:
    """Read the remote source receipt declared by an isolated incumbent.

    The worker independently recomputes these values before it constructs a
    policy.  This helper only freezes the expected receipt in the host's
    immutable experiment contract.
    """

    commit = str(raw.get("source_commit", ""))
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise HeadToHeadError(
            "isolated incumbent.source_commit must be a full lowercase Git commit id."
        )
    tree_sha256 = _required_sha256(
        raw.get("source_tree_sha256"), label="isolated incumbent.source_tree_sha256"
    )
    fingerprint = str(raw.get("engine_fingerprint", ""))
    if not fingerprint:
        raise HeadToHeadError("isolated incumbent.engine_fingerprint must be non-empty.")
    return commit, tree_sha256, fingerprint


def _validate_isolated_receipt(
    receipt: object,
    *,
    incumbent: MctsPolicySpec,
    bootstrap_sha256: str,
) -> Mapping[str, Any]:
    """Check the durable child receipt before a game is accepted or resumed."""

    payload = _mapping(receipt, label="isolated worker receipt")
    if dict(_mapping(payload.get("policy"), label="isolated worker receipt.policy")) != (
        incumbent.to_payload()
    ):
        raise HeadToHeadError(
            "isolated worker receipt policy does not exactly match the declared incumbent."
        )
    if payload.get("commit") != incumbent.source_commit:
        raise HeadToHeadError("isolated worker receipt source commit differs from incumbent.")
    if payload.get("tree_sha256") != incumbent.source_tree_sha256:
        raise HeadToHeadError("isolated worker receipt source tree differs from incumbent.")
    if payload.get("tree_status") != "clean_tracked_checkout":
        raise HeadToHeadError("isolated worker receipt does not attest a clean source checkout.")
    if payload.get("engine_fingerprint") != incumbent.engine_fingerprint:
        raise HeadToHeadError("isolated worker receipt engine differs from incumbent.")
    if payload.get("worker_bootstrap_sha256") != bootstrap_sha256:
        raise HeadToHeadError(
            "isolated worker receipt bootstrap differs from the host's declared adapter."
        )
    return dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--manifest", required=True, help="frozen MCTS-vs-MCTS experiment JSON")
    parser.add_argument("--out-dir", required=True, help="new or exact-resume durable result directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--isolated-incumbent-source-root",
        help=(
            "clean historical PokeZero checkout for the incumbent; enables the only "
            "source-different execution mode"
        ),
    )
    parser.add_argument(
        "--isolated-incumbent-python",
        help="Python executable from the isolated incumbent build (required with its source root)",
    )
    parser.add_argument(
        "--isolated-worker-timeout-seconds",
        type=float,
        default=60.0,
        help="per-decision response deadline for the isolated incumbent worker",
    )
    parser.add_argument("--skip-build-check", action="store_true", help="dry inspection only; never scored")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.skip_build_check:
        raise HeadToHeadError("--skip-build-check is not permitted for an MCTS-vs-MCTS result.")
    isolated_values = (
        args.isolated_incumbent_source_root,
        args.isolated_incumbent_python,
    )
    if any(isolated_values) and not all(isolated_values):
        raise HeadToHeadError(
            "--isolated-incumbent-source-root and --isolated-incumbent-python must be "
            "provided together."
        )
    if args.isolated_worker_timeout_seconds <= 0:
        raise HeadToHeadError("--isolated-worker-timeout-seconds must be positive.")
    execution_mode = "isolated_build" if all(isolated_values) else "in_process"
    out_root = _durable_output_root(args.out_dir)
    manifest = _load_manifest(args.manifest)
    declared_showdown_source_sha256 = _declared_showdown_source_sha256(manifest)
    declared_source_tree_sha256 = _declared_source_tree_sha256(manifest)
    seeds = _seeds(manifest)
    bootstrap = _mapping(manifest.get("bootstrap"), label="manifest.bootstrap")
    resamples = int(bootstrap.get("resamples", 0))
    bootstrap_seed = int(bootstrap.get("seed", -1))
    if resamples <= 0 or bootstrap_seed < 0:
        raise HeadToHeadError("manifest.bootstrap requires positive resamples and a non-negative seed.")
    max_decision_rounds = int(manifest.get("max_decision_rounds", 0))
    if max_decision_rounds <= 0:
        raise HeadToHeadError("manifest.max_decision_rounds must be positive.")

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from engine_build_fingerprint import assert_fresh, compute_fingerprint  # noqa: PLC0415
    from pokezero.dex import load_showdown_dex_cached  # noqa: PLC0415
    from pokezero.engine_search import (  # noqa: PLC0415
        EngineMctsPolicy,
        EnvTier2AnnotationSource,
    )
    from pokezero.local_showdown import (  # noqa: PLC0415
        LocalShowdownConfig,
        LocalShowdownEnv,
        env_config_from_checkpoint_provenance,
    )
    from pokezero.mcts_eval.lattice import materialize_search_artifacts  # noqa: PLC0415
    from pokezero.mcts_eval.isolated_policy import (  # noqa: PLC0415
        IsolatedMctsPolicy,
        IsolatedPolicyLaunch,
    )
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract  # noqa: PLC0415
    from pokezero.neural_policy import (  # noqa: PLC0415
        category_vocab_from_model_config,
        feature_masks_from_model_config,
        load_transformer_model_config,
        observation_spec_from_model_config,
    )
    from pokezero.randbat import load_gen3_randbat_source_cached  # noqa: PLC0415
    from pokezero.rollout import RolloutConfig, RolloutDriver  # noqa: PLC0415

    assert_fresh()
    source = _source_provenance()
    source_commit = source["commit"]
    source_tree_sha256 = source["tree_sha256"]
    if declared_source_tree_sha256 != source_tree_sha256:
        raise HeadToHeadError(
            "manifest source-tree identity does not match the executing source/image receipt: "
            f"declared {declared_source_tree_sha256}, active {source_tree_sha256}."
        )
    engine_fingerprint = str(compute_fingerprint()["fingerprint"])
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    showdown_source = _showdown_source_provenance(args.showdown_root)
    showdown_source_sha256 = str(showdown_source["content_sha256"])
    if declared_showdown_source_sha256 != showdown_source_sha256:
        raise HeadToHeadError(
            "manifest Showdown source identity does not match --showdown-root: "
            f"declared {declared_showdown_source_sha256}, active {showdown_source_sha256}."
        )
    contract = resolve_checkpoint_contract(
        args.checkpoint,
        model_device=args.device,
        showdown_root=args.showdown_root,
        showdown_source_sha256=showdown_source_sha256,
        expected_showdown_source_sha256=declared_showdown_source_sha256,
    )
    artifacts = materialize_search_artifacts(contract, showdown_root=args.showdown_root)
    candidate, candidate_config = _runtime_spec(
        _mapping(manifest.get("candidate"), label="manifest.candidate"),
        role="candidate",
        checkpoint=args.checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        source_commit=source_commit,
        source_tree_sha256=source_tree_sha256,
        engine_fingerprint=engine_fingerprint,
        showdown_source_sha256=showdown_source_sha256,
        model_path=artifacts["model_path"],
        tables_path=artifacts["tables_path"],
        device=args.device,
    )
    incumbent_raw = _mapping(manifest.get("incumbent"), label="manifest.incumbent")
    isolated_source_root: Path | None = None
    isolated_bootstrap_sha256: str | None = None
    if execution_mode == "isolated_build":
        isolated_source_root = Path(args.isolated_incumbent_source_root).expanduser().resolve()
        if not isolated_source_root.is_dir():
            raise HeadToHeadError(
                "--isolated-incumbent-source-root does not name an existing directory."
            )
        remote_commit, remote_tree_sha256, remote_fingerprint = _isolated_incumbent_identity(
            incumbent_raw
        )
        incumbent, incumbent_config = _runtime_spec(
            incumbent_raw,
            role="isolated incumbent",
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            source_commit=remote_commit,
            source_tree_sha256=remote_tree_sha256,
            engine_fingerprint=remote_fingerprint,
            showdown_source_sha256=showdown_source_sha256,
            model_path=artifacts["model_path"],
            tables_path=artifacts["tables_path"],
            device=args.device,
        )
        isolated_bootstrap_sha256 = _sha256_file(
            REPO_ROOT / "scripts" / "mcts_isolated_policy_worker.py"
        )
    else:
        incumbent, incumbent_config = _runtime_spec(
            incumbent_raw,
            role="incumbent",
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            source_commit=source_commit,
            source_tree_sha256=source_tree_sha256,
            engine_fingerprint=engine_fingerprint,
            showdown_source_sha256=showdown_source_sha256,
            model_path=artifacts["model_path"],
            tables_path=artifacts["tables_path"],
            device=args.device,
        )

    model_config = load_transformer_model_config(args.checkpoint)
    vocabulary = category_vocab_from_model_config(model_config, args.showdown_root)
    env_config = env_config_from_checkpoint_provenance(
        LocalShowdownConfig(
            showdown_root=args.showdown_root,
            set_belief_source=True,
            category_vocab=vocabulary,
        ),
        feature_masks_from_model_config(model_config),
        required_specs=observation_spec_from_model_config(model_config),
        required_vocabs=vocabulary,
        context="MCTS-vs-MCTS paired runner",
    )
    dex = load_showdown_dex_cached(args.showdown_root)
    set_source = load_gen3_randbat_source_cached(args.showdown_root)

    _write_immutable_json(
        out_root / "manifest.json",
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "declared_manifest": manifest,
            "active_source_commit": source_commit,
            "active_source": source,
            "active_engine_fingerprint": engine_fingerprint,
            "active_checkpoint_sha256": checkpoint_sha256,
            "active_showdown_source": showdown_source,
            "candidate": candidate.to_payload(),
            "incumbent": incumbent.to_payload(),
            "seeds": list(seeds),
            "execution_mode": execution_mode,
            "isolated_incumbent": (
                {
                    "source_root": str(isolated_source_root),
                    "python": str(args.isolated_incumbent_python),
                    "worker_bootstrap_sha256": isolated_bootstrap_sha256,
                }
                if execution_mode == "isolated_build"
                else None
            ),
        },
    )

    isolated_workers: dict[tuple[int, str], IsolatedMctsPolicy] = {}

    def session_factory(seed: int, candidate_seat: str):
        env = LocalShowdownEnv(env_config)
        annotations = EnvTier2AnnotationSource(env)
        candidate_policy = PublicOnlyMctsPolicy(
            EngineMctsPolicy(
                dex=dex,
                set_source=set_source,
                config=candidate_config,
                policy_id=candidate.policy_id,
                annotation_source=annotations,
            )
        )
        if execution_mode == "isolated_build":
            assert isolated_source_root is not None
            assert isolated_bootstrap_sha256 is not None
            isolated = IsolatedMctsPolicy(
                IsolatedPolicyLaunch(
                    policy=incumbent,
                    command=(
                        str(args.isolated_incumbent_python),
                        str(REPO_ROOT / "scripts" / "mcts_isolated_policy_worker.py"),
                    ),
                    worker_config={
                        "source_root": str(isolated_source_root),
                        "showdown_root": str(Path(args.showdown_root).expanduser().resolve()),
                        "worker_bootstrap_sha256": isolated_bootstrap_sha256,
                    },
                    response_timeout_seconds=args.isolated_worker_timeout_seconds,
                    stderr_path=(
                        out_root
                        / "worker-stderr"
                        / f"seed-{seed}-{candidate_seat}-{os.getpid()}.log"
                    ),
                    working_directory=isolated_source_root,
                ),
                annotation_source=annotations,
            )
            isolated_workers[(seed, candidate_seat)] = isolated
            incumbent_policy = PublicOnlyMctsPolicy(isolated)
        else:
            incumbent_policy = PublicOnlyMctsPolicy(
                EngineMctsPolicy(
                    dex=dex,
                    set_source=set_source,
                    config=incumbent_config,
                    policy_id=incumbent.policy_id,
                    annotation_source=annotations,
                )
            )
        other_seat = "p2" if candidate_seat == "p1" else "p1"
        driver = RolloutDriver(
            env=env,
            policies={candidate_seat: candidate_policy, other_seat: incumbent_policy},
            config=RolloutConfig(
                max_decision_rounds=max_decision_rounds,
                format_id="gen3randombattle",
                record_policy_timing=True,
                hide_opponent_legal_action_masks=True,
            ),
        )
        return driver, candidate_policy, incumbent_policy

    all_games = []
    for seed in seeds:
        completed = load_pair(
            out_root, seed=seed, candidate=candidate, incumbent=incumbent
        )
        if execution_mode == "isolated_build":
            assert isolated_bootstrap_sha256 is not None
            for candidate_seat in completed:
                receipt_path = (
                    out_root
                    / "worker-receipts"
                    / f"seed-{seed}-{candidate_seat}-incumbent.json"
                )
                try:
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise HeadToHeadError(
                        "completed isolated game has no readable matching worker receipt "
                        f"at {receipt_path}: {error}"
                    ) from error
                _validate_isolated_receipt(
                    receipt,
                    incumbent=incumbent,
                    bootstrap_sha256=isolated_bootstrap_sha256,
                )

        def on_game(game):
            if execution_mode == "isolated_build":
                assert isolated_bootstrap_sha256 is not None
                isolated = isolated_workers.pop((game.seed, game.candidate_seat), None)
                if isolated is None:
                    raise HeadToHeadError(
                        "isolated worker was not retained for its completed game receipt."
                    )
                receipt = _validate_isolated_receipt(
                    isolated.worker_receipt,
                    incumbent=incumbent,
                    bootstrap_sha256=isolated_bootstrap_sha256,
                )
                _write_immutable_json(
                    out_root
                    / "worker-receipts"
                    / f"seed-{game.seed}-{game.candidate_seat}-incumbent.json",
                    receipt,
                )
            write_game_immutable(out_root, game)

        games = play_mirrored_pair(
            seed=seed,
            candidate=candidate,
            incumbent=incumbent,
            candidate_factory=lambda: None,
            incumbent_factory=lambda: None,
            driver_factory=lambda *_: None,
            completed=completed,
            session_factory=session_factory,
            on_game=on_game,
            execution_mode=execution_mode,
        )
        complete_pair(games, seed=seed, candidate=candidate, incumbent=incumbent)
        all_games.extend(games)
        print(f"completed mirrored pair seed={seed}", flush=True)

    summary = summarize_complete_pairs(
        all_games,
        seeds=seeds,
        candidate=candidate,
        incumbent=incumbent,
        bootstrap_resamples=resamples,
        bootstrap_seed=bootstrap_seed,
    )
    _write_immutable_json(out_root / "summary.json", summary)
    _write_immutable_json(
        out_root / "COMPLETE.json",
        {
            "schema_version": COMPLETE_SCHEMA_VERSION,
            "summary_sha256": _sha256_file(out_root / "summary.json"),
            "games": len(all_games),
            "pairs": len(seeds),
            "candidate_provenance_sha256": candidate.provenance_sha256,
            "incumbent_provenance_sha256": incumbent.provenance_sha256,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HeadToHeadError as error:
        print(f"MCTS-vs-MCTS REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
