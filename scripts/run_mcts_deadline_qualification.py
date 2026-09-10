#!/usr/bin/env python3
"""Run the bounded, source-bound model-MCTS deadline qualification.

This is a measurement gate, not a game trial.  It replays the pre-frozen timing
corpus with one fixed model decision budget, writes atomic per-decision evidence,
and emits PASS only when the durable evidence proves a nonzero native deadline
prefix without fallbacks, invalid choices, refused native calls, or zero-work
decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

# The bounded runner may advance independently from the deadline mechanism it
# qualifies.  It therefore records and checks the reviewed engine source and
# installed-build fingerprint separately from the runner image's own receipt.
SOURCE_RECEIPT_SCHEMA_VERSION = "pokezero.mcts-deadline-source-receipt.v1"
# The runner can advance independently, but a qualification is only about the
# deadline mechanism reviewed at this source point.  These values are a second,
# local guard in addition to the image receipt and installed-build fingerprint.
REVIEWED_DEADLINE_SOURCE_COMMIT = "8609d301399738a8081f0e938a3cc4ed7d39abdd"
REVIEWED_ENGINE_SEARCH_SHA256 = "8d647f13440173cf939ea36c7d5940e64544388fb82b0315d2f67c9a9b845eb5"
REVIEWED_ENGINE_FINGERPRINT = "82201ace3c55a9a0e04ccf1085342155cca8a5b38f939aba11745f1bf513ae6e"
REQUIRED_RECEIPT_FILES = (
    "scripts/run_mcts_deadline_qualification.py",
    "scripts/engine_build_fingerprint.py",
    "src/pokezero/engine_search.py",
    "src/pokezero/mcts_eval/deadline_qualification.py",
    "src/pokezero/mcts_eval/lattice.py",
)

from pokezero.mcts_eval.deadline_qualification import (  # noqa: E402
    DEADLINE_QUALIFICATION_SCHEMA_VERSION,
    DeadlineQualificationError,
    DeadlineQualificationRequirements,
    validate_deadline_decision,
    validate_deadline_qualification,
)
from pokezero.mcts_eval.lattice import _LiveEngineTimingDecider  # noqa: E402
from pokezero.mcts_eval.manifest import SearchConfig  # noqa: E402
from pokezero.mcts_eval.resolver import resolve_checkpoint_contract, sha256_file  # noqa: E402
from pokezero.mcts_eval.timing_corpus import (  # noqa: E402
    CorpusError,
    canonical_json_sha256,
    read_corpus,
    validate_representative_timing_panel,
)
from engine_build_fingerprint import assert_fresh, compute_fingerprint  # noqa: E402


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise DeadlineQualificationError(f"{path}: expected a JSON object")
    return value


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash_source_files(repo_root: Path, paths: Sequence[Path]) -> str:
    """Hash each execution input with its stable repository-relative name."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(repo_root).as_posix()):
        relative = path.relative_to(repo_root).as_posix()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _execution_source_files() -> list[Path]:
    """All Python, bridge, and native code this replay can actually execute."""

    paths: list[Path] = []
    for root, patterns in (
        (ROOT / "src" / "pokezero", ("*.py",)),
        (ROOT / "scripts", ("*.py", "*.mjs")),
        (ROOT / "rust" / "pokezero-search", ("*",)),
    ):
        if not root.is_dir():
            continue
        for pattern in patterns:
            paths.extend(
                path
                for path in root.rglob(pattern)
                if path.is_file()
                and "__pycache__" not in path.parts
                and "target" not in path.parts
            )
    pyproject = ROOT / "pyproject.toml"
    if pyproject.is_file():
        paths.append(pyproject)
    return sorted(set(paths))


def _active_source_provenance() -> dict[str, Any]:
    """Bind source by contents; images without a .git directory are supported."""

    stamped = os.environ.get("POKEZERO_COMMIT", "").strip().lower()
    source_files = _execution_source_files()
    if not source_files:
        raise DeadlineQualificationError("cannot hash the active qualification source tree")
    git_metadata = ROOT / ".git"
    if git_metadata.exists():
        try:
            commit = subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().lower()
            dirty = subprocess.run(
                ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise DeadlineQualificationError("cannot inspect source checkout provenance") from error
        if not _is_lower_hex(commit, 40):
            raise DeadlineQualificationError("source checkout HEAD is not a full lowercase Git commit")
        if stamped and stamped != commit:
            raise DeadlineQualificationError(
                "POKEZERO_COMMIT does not match the executing source checkout"
            )
        if dirty:
            raise DeadlineQualificationError("source checkout is dirty; refusing an unbound replay")
        status = "clean_git_checkout"
    else:
        if not _is_lower_hex(stamped, 40):
            raise DeadlineQualificationError(
                "an image without .git must set POKEZERO_COMMIT to its full lowercase source commit"
            )
        commit = stamped
        status = "explicit_commit_without_git"
    return {
        "commit": commit,
        "execution_tree_sha256": _hash_source_files(ROOT, source_files),
        "tree_status": status,
    }


def _receipt_path(path: str) -> Path:
    receipt_path = Path(path).expanduser().resolve()
    if not receipt_path.is_file():
        raise DeadlineQualificationError(f"source receipt not found: {receipt_path}")
    return receipt_path


def _source_receipt(path: str) -> dict[str, Any]:
    """Load and fail-close validate the image-builder's immutable source receipt."""

    receipt = dict(_read_json(_receipt_path(path)))
    if receipt.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION:
        raise DeadlineQualificationError("source receipt schema is not supported")
    if receipt.get("complete") is not True:
        raise DeadlineQualificationError("source receipt is not marked complete")
    image = receipt.get("immutable_image")
    if not isinstance(image, str) or "@sha256:" not in image:
        raise DeadlineQualificationError("source receipt must declare a digest-qualified immutable image")
    digest = image.rsplit("@sha256:", 1)[-1]
    if not _is_lower_hex(digest, 64):
        raise DeadlineQualificationError("source receipt immutable image has an invalid digest")
    if not _is_lower_hex(receipt.get("source_commit"), 40):
        raise DeadlineQualificationError("source receipt source_commit must be a full lowercase Git commit")
    if not _is_lower_hex(receipt.get("execution_tree_sha256"), 64):
        raise DeadlineQualificationError("source receipt execution_tree_sha256 must be a lowercase SHA-256")
    if not _is_lower_hex(receipt.get("engine_fingerprint"), 64):
        raise DeadlineQualificationError("source receipt engine_fingerprint must be a lowercase SHA-256")
    declared_files = receipt.get("source_files_sha256")
    if not isinstance(declared_files, Mapping):
        raise DeadlineQualificationError("source receipt source_files_sha256 must be an object")
    if any(path not in declared_files for path in REQUIRED_RECEIPT_FILES):
        raise DeadlineQualificationError("source receipt omits a required executable source file")
    for relative, expected in declared_files.items():
        if not isinstance(relative, str) or not _is_lower_hex(expected, 64):
            raise DeadlineQualificationError("source receipt has an invalid source file hash")
        candidate = (ROOT / relative).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError as error:
            raise DeadlineQualificationError("source receipt source file escapes the repository") from error
        if not candidate.is_file():
            raise DeadlineQualificationError(f"source receipt file is absent at runtime: {relative}")
        if sha256_file(candidate) != expected:
            raise DeadlineQualificationError(f"source receipt file drift: {relative}")
    return receipt


def _verify_source_receipt(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _source_receipt(path)
    active = _active_source_provenance()
    if active["commit"] != receipt["source_commit"]:
        raise DeadlineQualificationError("active source commit differs from the immutable image receipt")
    if active["execution_tree_sha256"] != receipt["execution_tree_sha256"]:
        raise DeadlineQualificationError("active execution source differs from the immutable image receipt")
    return receipt, active


def _safe_decision_name(decision_id: str, index: int) -> str:
    digest = hashlib.sha256(decision_id.encode("utf-8")).hexdigest()[:16]
    return f"{index:02d}-{digest}.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--expected-corpus-sha256", required=True)
    parser.add_argument("--expected-corpus-file-sha256", required=True)
    parser.add_argument(
        "--source-receipt",
        required=True,
        help="Immutable source-image receipt mounted beside the exact image being run.",
    )
    parser.add_argument("--expected-showdown-source-sha256", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--deadline-ms", type=int, default=1_000)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--sims", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only independently revalidated, durable decision units for this exact manifest.",
    )
    return parser.parse_args(argv)


def _frozen_manifest(
    *,
    args: argparse.Namespace,
    source_receipt: Mapping[str, Any],
    active_source: Mapping[str, Any],
    corpus_manifest: Any,
    corpus_file_sha256: str,
    checkpoint_contract: Any,
    deadline_mechanics: Mapping[str, Any],
    showdown_source: Mapping[str, Any],
    requirements: DeadlineQualificationRequirements,
) -> dict[str, Any]:
    return {
        "schema_version": DEADLINE_QUALIFICATION_SCHEMA_VERSION,
        "source_receipt": dict(source_receipt),
        "active_source": dict(active_source),
        "corpus_path": str(Path(args.corpus).resolve()),
        "corpus_sha256": corpus_manifest.corpus_sha256,
        "corpus_file_sha256": corpus_file_sha256,
        "checkpoint": checkpoint_contract.to_manifest(),
        "deadline_mechanics": dict(deadline_mechanics),
        "showdown_source": dict(showdown_source),
        "requirements": requirements.to_payload(),
        "search_config": {
            "depth": args.depth,
            "sims": args.sims,
            "batch": args.batch,
            "worlds": args.worlds,
            "early_stop": False,
            # This is a deadline-mechanics qualification, not a prior-policy
            # study.  Freeze both selection-prior toggles off exactly as the
            # predeclared contract requires.
            "model_priors": False,
            "use_opponent_priors": False,
            "model_decision_time_ms": args.deadline_ms,
        },
    }


def _showdown_dependency_paths(root: Path) -> list[Path]:
    """Return every built Showdown byte the Gen 3 runtime can load."""

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
        raise DeadlineQualificationError(
            "cannot bind Showdown runtime; required input is missing: "
            f"{missing[0].relative_to(root)}"
        )
    paths = set(required)
    paths.update((root / "dist").rglob("*.js"))
    paths.update((root / "dist").rglob("*.json"))
    return sorted(path for path in paths if path.is_file())


def _showdown_source_provenance(showdown_root: str | Path) -> dict[str, Any]:
    """Bind all content loaded by Showdown and require its checkout to be clean."""

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
        ).stdout.strip().lower()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise DeadlineQualificationError(f"cannot bind Showdown runtime identity from {root}") from error
    if top_level != root:
        raise DeadlineQualificationError("--showdown-root must be the root of its Showdown Git checkout")
    if not _is_lower_hex(commit, 40):
        raise DeadlineQualificationError("Showdown HEAD is not a full lowercase Git commit")
    if dirty:
        raise DeadlineQualificationError("Showdown checkout is dirty; refusing an unbound replay")
    digest = hashlib.sha256()
    for path in _showdown_dependency_paths(root):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return {"content_sha256": digest.hexdigest(), "git_commit": commit, "git_clean": True}


def _verify_showdown_source(
    showdown_root: str | Path, expected_showdown_source_sha256: str
) -> dict[str, Any]:
    if not _is_lower_hex(expected_showdown_source_sha256, 64):
        raise DeadlineQualificationError("expected Showdown source SHA-256 must be lowercase hex")
    source = _showdown_source_provenance(showdown_root)
    if source["content_sha256"] != expected_showdown_source_sha256:
        raise DeadlineQualificationError("active Showdown runtime differs from the frozen source hash")
    return source


def _deadline_mechanics_evidence(source_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a source/image pair whose installed native engine is stale or different."""

    engine_search_path = ROOT / "src" / "pokezero" / "engine_search.py"
    engine_search_sha256 = sha256_file(engine_search_path)
    if engine_search_sha256 != REVIEWED_ENGINE_SEARCH_SHA256:
        raise DeadlineQualificationError(
            "engine_search.py differs from the reviewed deadline mechanism"
        )
    declared_files = source_receipt["source_files_sha256"]
    if declared_files["src/pokezero/engine_search.py"] != engine_search_sha256:
        raise DeadlineQualificationError("source receipt engine_search.py hash differs from active source")
    try:
        assert_fresh()
        fingerprint = compute_fingerprint()
    except BaseException as error:  # assert_fresh exits with SystemExit on stale native artifacts.
        raise DeadlineQualificationError("installed native engine failed its freshness check") from error
    active_fingerprint = fingerprint.get("fingerprint")
    if active_fingerprint != REVIEWED_ENGINE_FINGERPRINT:
        raise DeadlineQualificationError("native engine fingerprint differs from the reviewed deadline mechanism")
    if active_fingerprint != source_receipt["engine_fingerprint"]:
        raise DeadlineQualificationError("native engine fingerprint differs from the immutable image receipt")
    return {
        "reviewed_source_commit": REVIEWED_DEADLINE_SOURCE_COMMIT,
        "engine_search_sha256": engine_search_sha256,
        "engine_build_fingerprint": dict(fingerprint),
    }


def _prepare_root(out_root: Path, manifest: Mapping[str, Any], *, resume: bool) -> None:
    manifest_path = out_root / "MANIFEST.json"
    terminals = [out_root / name for name in ("PASS.json", "NONPASS.json") if (out_root / name).exists()]
    if terminals:
        raise DeadlineQualificationError(
            f"{out_root}: already has terminal artifact(s): {', '.join(str(path.name) for path in terminals)}"
        )
    if manifest_path.exists():
        if not resume:
            raise DeadlineQualificationError(f"{out_root}: exists; pass --resume to revalidate durable units")
        if _read_json(manifest_path) != manifest:
            raise DeadlineQualificationError(f"{out_root}: existing manifest differs from this frozen contract")
    else:
        if out_root.exists() and any(out_root.iterdir()):
            raise DeadlineQualificationError(
                f"{out_root}: nonempty root has no manifest and cannot be adopted"
            )
        _atomic_json(manifest_path, manifest)
    _atomic_json(out_root / "RUNNING.json", {"manifest": manifest, "state": "RUNNING"})


def _decision_payload(
    *,
    record: Any,
    telemetry: Mapping[str, Any],
    outer_wall_ms: float,
) -> dict[str, Any]:
    engine = telemetry.get("engine_mcts")
    if not isinstance(engine, Mapping):
        engine = {}
    return {
        "decision_id": record.decision_id,
        "corpus_record_sha256": canonical_json_sha256(record.to_payload()),
        "battle_id": record.battle_id,
        "seat": record.seat,
        "turn_index": record.turn_index,
        "root_action": telemetry.get("root_action"),
        "outer_wall_ms": round(outer_wall_ms, 3),
        "invalid_actions": telemetry.get("invalid_actions"),
        "engine_mcts": dict(engine),
    }


def _validate_reused_decision(
    payload: Mapping[str, Any],
    *,
    record: Any,
    target: Path,
    requirements: DeadlineQualificationRequirements,
) -> dict[str, Any]:
    """Accept a resumable unit only if it is exactly the frozen corpus record."""

    if payload.get("decision_id") != record.decision_id:
        raise DeadlineQualificationError(f"{target}: durable unit belongs to a different corpus decision")
    if payload.get("corpus_record_sha256") != canonical_json_sha256(record.to_payload()):
        raise DeadlineQualificationError(
            f"{target}: durable unit identity differs from the frozen corpus record"
        )
    return validate_deadline_decision(payload, requirements=requirements)


def _run(args: argparse.Namespace, *, ownership: dict[str, bool]) -> dict[str, Any]:
    source_receipt, active_source = _verify_source_receipt(args.source_receipt)
    deadline_mechanics = _deadline_mechanics_evidence(source_receipt)
    showdown_source = _verify_showdown_source(
        args.showdown_root, args.expected_showdown_source_sha256
    )
    requirements = DeadlineQualificationRequirements(
        requested_ms=args.deadline_ms,
        sims_per_world=args.sims,
        worlds=args.worlds,
        expected_decisions=16,
    )
    corpus_file_sha256 = sha256_file(args.corpus)
    if corpus_file_sha256 != args.expected_corpus_file_sha256:
        raise DeadlineQualificationError(
            "raw corpus SHA-256 differs from the frozen corpus file"
        )
    try:
        corpus_manifest, records = read_corpus(args.corpus)
        coverage = validate_representative_timing_panel(records)
    except CorpusError as error:
        raise DeadlineQualificationError(f"timing corpus rejected: {error}") from error
    if corpus_manifest.corpus_sha256 != args.expected_corpus_sha256:
        raise DeadlineQualificationError("canonical corpus SHA-256 differs from the frozen corpus")
    if len(records) != requirements.expected_decisions:
        raise DeadlineQualificationError(
            f"corpus has {len(records)} decisions; qualification requires {requirements.expected_decisions}"
        )
    checkpoint_contract = resolve_checkpoint_contract(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        model_device=args.model_device,
        showdown_root=args.showdown_root,
        showdown_source_sha256=showdown_source["content_sha256"],
        expected_showdown_source_sha256=args.expected_showdown_source_sha256,
    )
    manifest = _frozen_manifest(
        args=args,
        source_receipt=source_receipt,
        active_source=active_source,
        corpus_manifest=corpus_manifest,
        corpus_file_sha256=corpus_file_sha256,
        checkpoint_contract=checkpoint_contract,
        deadline_mechanics=deadline_mechanics,
        showdown_source=showdown_source,
        requirements=requirements,
    )
    out_root = Path(args.out_root)
    _prepare_root(out_root, manifest, resume=args.resume)
    ownership["verified"] = True
    _atomic_json(out_root / "CORPUS_COVERAGE.json", coverage)

    config = SearchConfig(depth=args.depth, sims=args.sims, batch=args.batch, worlds=args.worlds)
    rows: list[Mapping[str, Any]] = []
    decider = _LiveEngineTimingDecider(
        checkpoint_contract,
        args.showdown_root,
        model_decision_time_ms=args.deadline_ms,
        model_priors=False,
        use_opponent_priors=False,
    )
    try:
        for index, record in enumerate(records):
            target = out_root / "decisions" / _safe_decision_name(record.decision_id, index)
            if target.exists():
                payload = _read_json(target)
                validated = _validate_reused_decision(
                    payload, record=record, target=target, requirements=requirements
                )
                rows.append(payload)
                print(f"reused {record.decision_id}: {validated['native_prefixes']} native prefixes", flush=True)
                continue
            payload: dict[str, Any] | None = None
            try:
                timed = decider.prepare(record, config)
                started = time.perf_counter()
                telemetry = timed()
                payload = _decision_payload(
                    record=record,
                    telemetry=telemetry,
                    outer_wall_ms=(time.perf_counter() - started) * 1000.0,
                )
                validated = validate_deadline_decision(payload, requirements=requirements)
            except Exception as error:  # noqa: BLE001 - preserve the failed live witness too
                _atomic_json(
                    out_root / "failures" / _safe_decision_name(record.decision_id, index),
                    {
                        "decision_id": record.decision_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "payload": payload,
                    },
                )
                raise
            assert payload is not None
            _atomic_json(target, payload)
            rows.append(payload)
            print(f"completed {record.decision_id}: {validated['native_prefixes']} native prefixes", flush=True)
    finally:
        decider.close()
    summary = validate_deadline_qualification(rows, requirements=requirements)
    return {"manifest": manifest, "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_root = Path(args.out_root)
    # A new/empty root is ours to terminalize even when preflight itself fails.
    # A resume root becomes ours only *after* its immutable manifest has matched;
    # this prevents a mistaken invocation from appending a NONPASS to somebody
    # else's in-flight or terminal study.
    new_or_empty_root = not out_root.exists() or not any(out_root.iterdir())
    ownership = {"verified": new_or_empty_root}
    if not new_or_empty_root and not args.resume:
        print(f"NONPASS: {out_root}: exists; pass --resume to revalidate durable units", file=sys.stderr)
        return 2
    try:
        result = _run(args, ownership=ownership)
    except Exception as error:  # noqa: BLE001 - terminal diagnostics must survive every failure
        if ownership["verified"]:
            out_root.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                out_root / "NONPASS.json",
                {
                    "schema_version": DEADLINE_QUALIFICATION_SCHEMA_VERSION,
                    "state": "NONPASS",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            (out_root / "RUNNING.json").unlink(missing_ok=True)
        print(f"NONPASS: {error}", file=sys.stderr)
        return 2
    _atomic_json(
        out_root / "PASS.json",
        {
            "schema_version": DEADLINE_QUALIFICATION_SCHEMA_VERSION,
            "state": "PASS",
            **result,
        },
    )
    _atomic_json(
        out_root / "DEADLINE_QUALIFICATION_PASS.json",
        {
            "schema_version": DEADLINE_QUALIFICATION_SCHEMA_VERSION,
            "state": "PASS",
            "marker": "DEADLINE_QUALIFICATION_PASS",
        },
    )
    (out_root / "RUNNING.json").unlink(missing_ok=True)
    print("DEADLINE QUALIFICATION PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
