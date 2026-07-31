#!/usr/bin/env python3
"""Verify C26's immutable public provenance and current regression surface.

C26 retained no replay rows or classifier output. This checker keeps the
original public merge as immutable provenance while exercising the current
checkout's switch-prefixed renderer contract. It cannot prove a fresh result
for any historical identity or grant certification clearance.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path


PUBLIC_MERGE = "8af4f42e99ef9b6a0b809027976a27a8d135cd3c"
EVENTS = "rust/pokezero-search/src/events.rs"
NATIVE_TEST = "rust/pokezero-search/tests/gen3_confusion_event_renderer.rs"
PREDICTION = "reports/c26_switch_confusion_event_attribution_prediction.md"
SUPERSESSION = "reports/c26_switch_confusion_event_attribution_supersession.md"
PATCH_LIST_SHA256 = "690b9407059c4a9322b9bee2a7dc59f3a5ea8477c5c7b493c8243b3157c903ea"
ENGINE_SOURCE_SPEC = {
    "schema": "pokezero-engine-upstream-source/1",
    "distribution": "poke-engine",
    "version": "0.0.47",
    "archive": "poke_engine-0.0.47.tar.gz",
    "sha256": "84a7dfad5ce4650a2cb9250999597c594385069eb33622c6a14bb1279694b434",
}
CURRENT_REGRESSION = "switch_prefixed_exact_self_hit_is_untagged_and_safe"
EXPECTED_RENDERER_TEST_COUNT = 22
CURRENT_PUBLIC_INPUTS = (
    EVENTS,
    "rust/pokezero-search/Cargo.toml",
    "third_party/poke-engine-base-source.json",
    "third_party/poke-engine-gen3-patches.txt",
)
_CARGO_RUNNING_TESTS = re.compile(r"(?m)^running (?P<count>\d+) tests?$")
_CARGO_RESULT = re.compile(
    r"(?m)^test result: ok\. (?P<passed>\d+) passed; (?P<failed>\d+) failed; "
    r"(?P<ignored>\d+) ignored; (?P<measured>\d+) measured; "
    r"(?P<filtered>\d+) filtered out;.*$"
)


def command(
    repo: Path,
    label: str,
    args: list[str],
    *,
    cwd: Path | None = None,
    forbid_skip: bool = False,
) -> str:
    """Run a required command and surface all failures rather than skipping a gate."""

    try:
        result = subprocess.run(
            args,
            cwd=cwd or repo,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as error:
        raise RuntimeError(f"{label} could not start: {error}") from error
    if result.returncode:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}: {' '.join(args)}\n"
            f"{result.stdout}"
        )
    if forbid_skip and "skipped" in result.stdout.lower():
        raise RuntimeError(f"{label} skipped required coverage:\n{result.stdout}")
    return result.stdout


def git(repo: Path, *args: str) -> str:
    return command(repo, f"git {' '.join(args)}", ["git", "-C", str(repo), *args])


def require(source: str, fragment: str, label: str) -> None:
    if fragment not in source:
        raise RuntimeError(f"missing historical {label}: {fragment!r}")


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def refresh_authoritative_origin_main(repo: Path) -> str:
    """Fetch origin/main and return the exact commit used for public claims."""

    command(
        repo,
        "authoritative origin/main refresh",
        [
            "git",
            "-C",
            str(repo),
            "fetch",
            "--no-tags",
            "origin",
            "+refs/heads/main:refs/remotes/origin/main",
        ],
    )
    return git(repo, "rev-parse", "--verify", "origin/main^{commit}").strip()


def require_cargo_regression_evidence(stdout: str, test_name: str) -> dict[str, int | str]:
    """Require Cargo output proving the named regression ran without exclusions."""

    test_line = re.compile(rf"(?m)^test {re.escape(test_name)} \.\.\. ok$")
    if not test_line.search(stdout):
        raise RuntimeError(
            f"cargo output does not prove required regression {test_name!r} executed and passed:\n"
            f"{stdout}"
        )

    running = _CARGO_RUNNING_TESTS.findall(stdout)
    results = list(_CARGO_RESULT.finditer(stdout))
    if len(running) != 1 or len(results) != 1:
        raise RuntimeError(
            "cargo output must contain exactly one integration-test count and result summary:\n"
            f"{stdout}"
        )

    expected_count = int(running[0])
    summary = {name: int(value) for name, value in results[0].groupdict().items()}
    if expected_count <= 0:
        raise RuntimeError(f"cargo reported no runnable tests for required regression {test_name!r}")
    if summary["failed"] != 0:
        raise RuntimeError(f"cargo reported failed tests for required regression {test_name!r}")
    if summary["ignored"] != 0:
        raise RuntimeError(f"cargo reported ignored tests for required regression {test_name!r}")
    if summary["filtered"] != 0:
        raise RuntimeError(f"cargo filtered tests for required regression {test_name!r}")
    if expected_count != EXPECTED_RENDERER_TEST_COUNT:
        raise RuntimeError(
            "cargo renderer test count changed for required regression "
            f"{test_name!r}: expected {EXPECTED_RENDERER_TEST_COUNT}, got {expected_count}"
        )
    if summary["passed"] != expected_count:
        raise RuntimeError(
            "cargo passed-test count does not match the runnable integration-test count "
            f"for required regression {test_name!r}: {summary['passed']} != {expected_count}"
        )

    return {"test": test_name, "expected_count": expected_count, **summary}


def verify_historical_public_merge(repo: Path, authoritative_main: str) -> dict[str, object]:
    """Keep the original merge proof immutable and anchored to current main."""

    git(repo, "cat-file", "-e", f"{PUBLIC_MERGE}^{{commit}}")
    git(repo, "merge-base", "--is-ancestor", PUBLIC_MERGE, authoritative_main)
    parents = git(repo, "show", "-s", "--format=%P", PUBLIC_MERGE).split()
    if len(parents) != 2:
        raise RuntimeError(f"public implementation is not a merge commit: {parents!r}")

    public_events = git(repo, "show", f"{PUBLIC_MERGE}:{EVENTS}")
    public_tests = git(repo, "show", f"{PUBLIC_MERGE}:{NATIVE_TEST}")
    require(public_events, "fn confusion_self_hit_damage(", "damage derivation")
    require(public_events, "fn classify_confusion_self_hit(", "fail-closed classifier")
    require(
        public_events,
        "crash_matches || self_faint_move_can_be_self_only",
        "collision guard",
    )
    require(
        public_events,
        'out.lines.push(format!("|-damage|{ident}|{condition}"));',
        "untagged exact self-hit output",
    )
    require(
        public_tests,
        "fn recoil_after_an_executed_move_is_not_confusion_damage()",
        "ordinary-damage control",
    )
    return {"public_merge": PUBLIC_MERGE, "public_merge_parents": parents}


def verify_current_engine_inputs(repo: Path) -> dict[str, object]:
    """Authenticate tracked engine inputs and the Rust test's vendor target tree."""

    sys.path.insert(0, str(repo / "scripts"))
    import apply_poke_engine_patches as patch_stack
    import engine_build_fingerprint as engine_fingerprint
    import verify_poke_engine_source as source_verifier

    source_spec = source_verifier.source_pin()
    require_equal(source_spec, ENGINE_SOURCE_SPEC, "pinned engine source specification")
    patch_list_digest = hashlib.sha256(patch_stack.PATCH_LIST.read_bytes()).hexdigest()
    require_equal(patch_list_digest, PATCH_LIST_SHA256, "patch-list digest")

    fingerprint = engine_fingerprint.compute_fingerprint()
    require_equal(
        fingerprint["base_source"], ENGINE_SOURCE_SPEC, "engine fingerprint source specification"
    )
    patch_names = patch_stack.patch_names()
    require_equal(fingerprint["patches"], patch_names, "engine fingerprint patch order")
    patch_stack.patch_target_paths()

    vendored = engine_fingerprint.VENDORED
    if not vendored.is_dir():
        raise RuntimeError(f"missing vendored engine source required by Rust test: {vendored}")
    manifest = tomllib.loads((vendored / "Cargo.toml").read_text(encoding="utf-8"))
    require_equal(
        manifest.get("package", {}).get("version"),
        ENGINE_SOURCE_SPEC["version"],
        "vendored engine package version",
    )
    require_equal(
        patch_stack.patched_target_tree_sha256(vendored),
        patch_stack.PATCHED_TARGET_TREE_SHA256,
        "vendored patched target-tree digest",
    )
    return {
        "engine_version": ENGINE_SOURCE_SPEC["version"],
        "patch_count": len(patch_names),
        "patch_list_sha256": patch_list_digest,
        "patched_target_tree_sha256": patch_stack.PATCHED_TARGET_TREE_SHA256,
    }


def verify_current_regression_surface(repo: Path, authoritative_main: str) -> list[object]:
    """Exercise current-main-equivalent renderer behavior, not frozen source text."""

    git(repo, "merge-base", "--is-ancestor", authoritative_main, "HEAD")
    command(
        repo,
        "current checkout public-input comparison",
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--quiet",
            authoritative_main,
            "--",
            *CURRENT_PUBLIC_INPUTS,
        ],
    )
    command(
        repo,
        "C26 supersession verifier unit test",
        [
            "uv",
            "run",
            "--isolated",
            "--python",
            "3.12",
            "python",
            "tests/test_c26_switch_confusion_supersession.py",
        ],
        forbid_skip=True,
    )
    cargo_stdout = command(
        repo,
        "current switch-prefixed confusion renderer regression",
        [
            "cargo",
            "test",
            "--test",
            "gen3_confusion_event_renderer",
        ],
        cwd=repo / "rust" / "pokezero-search",
    )
    cargo_evidence = require_cargo_regression_evidence(cargo_stdout, CURRENT_REGRESSION)
    command(
        repo,
        "pinned poke-engine patch-stack test",
        [
            "uv",
            "run",
            "--isolated",
            "--python",
            "3.12",
            "python",
            "tests/test_poke_engine_patch_stack.py",
        ],
        forbid_skip=True,
    )
    command(
        repo,
        "public invariant test",
        [
            "uv",
            "run",
            "--isolated",
            "--python",
            "3.12",
            "python",
            "tests/test_public_invariant.py",
        ],
        forbid_skip=True,
    )
    return [
        "tests/test_c26_switch_confusion_supersession.py",
        {"cargo_renderer": cargo_evidence},
        "tests/test_poke_engine_patch_stack.py",
        "tests/test_public_invariant.py",
    ]


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    for path in (PREDICTION, SUPERSESSION):
        if not (repo / path).is_file():
            raise RuntimeError(f"missing C26 evidence record: {path}")

    authoritative_main = refresh_authoritative_origin_main(repo)
    historical = verify_historical_public_merge(repo, authoritative_main)
    engine = verify_current_engine_inputs(repo)
    current_checks = verify_current_regression_surface(repo, authoritative_main)
    evidence = {
        "verdict": "historical_public_merge_and_current_regression_verified",
        **historical,
        "authoritative_origin_main": authoritative_main,
        "current_checks": current_checks,
        "engine": engine,
        "identity_ledger": "not verified: retained labels are not independent provenance",
        "row_replay": "not asserted: no C26 row payload, checkpoint shard, or result artifact is retained in this repository",
        "clearance": "not granted: this is renderer-contract coverage, not a certification classifier replay",
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
