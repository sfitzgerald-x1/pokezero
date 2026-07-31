#!/usr/bin/env python
"""Certify the evidence-only C26 disposition against retained inputs.

The inputs are deliberately supplied at execution time. The committed readout
contains only stable opaque labels and SHA-256 digests, never archive locations
or retained payloads. A missing ``--shards`` argument is an execution refusal,
not a passing replay.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

READOUT_PATH = REPO_ROOT / "reports" / "c26_damage_composition_tail_readout.json"
MATCHER_PATH = "scripts/engine_transition_differential.py"
REGRESSION_IDENTITIES = ("2200760/86", "2300983/40", "2700145/92")


def _readout() -> dict[str, Any]:
    payload = json.loads(READOUT_PATH.read_text())
    if not isinstance(payload, dict):
        raise ValueError("C26 readout must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_archive_inputs(readout: Mapping[str, Any], shard_glob: str) -> list[Mapping[str, Any]]:
    from cert_sweep_reread import load_retained_rows

    archive = readout["pinned_baseline"]["retained_archive"]
    expected = archive["shards"]
    paths = [Path(path) for path in sorted(glob.glob(shard_glob))]
    if len(paths) != len(expected):
        raise ValueError("retained archive shard count does not match the pinned contract")
    for expected_shard, path in zip(expected, paths, strict=True):
        if _sha256(path) != expected_shard["sha256"]:
            raise ValueError(f"{expected_shard['label']}: SHA-256 does not match the pinned contract")
    try:
        return load_retained_rows(shard_glob, expected_rows=archive["population"])
    except ValueError as error:
        raise ValueError("retained archive does not satisfy the complete-population contract") from error


def _source_at_commit(commit: str) -> str:
    return subprocess.run(
        ["git", "show", f"{commit}:{MATCHER_PATH}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _module_from_source(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = f"{name}.py"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


@contextlib.contextmanager
def _using_matcher(module: types.ModuleType) -> Iterator[None]:
    previous = sys.modules.get("engine_transition_differential")
    sys.modules["engine_transition_differential"] = module
    try:
        yield
    finally:
        if previous is None:
            del sys.modules["engine_transition_differential"]
        else:
            sys.modules["engine_transition_differential"] = previous


def _reread(
    rows: list[Mapping[str, Any]], matcher: types.ModuleType
) -> tuple[dict[str, int], dict[str, dict[str, int | str]]]:
    from cert_sweep_reread import reread_row

    tally: Counter[str] = Counter()
    results: dict[str, dict[str, int | str]] = {}
    with _using_matcher(matcher):
        for row in rows:
            verdict, _misses, branches = reread_row(row)
            identity = f"{row['seed']}/{row['step']}"
            tally[verdict] += 1
            results[identity] = {"verdict": verdict, "branches": branches}
    return dict(tally), results


def _verdict_delta(
    baseline: Mapping[str, Mapping[str, int | str]],
    final: Mapping[str, Mapping[str, int | str]],
) -> dict[str, int]:
    if set(baseline) != set(final):
        raise ValueError("baseline and final rereads do not cover the same retained identities")
    delta = Counter()
    for identity, baseline_row in baseline.items():
        before = baseline_row["verdict"]
        after = final[identity]["verdict"]
        if before != after:
            delta[f"{before}_to_{after}"] += 1
    return dict(delta)


def _source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def _regression_ablation_check(
    rows: list[Mapping[str, Any]],
    *,
    baseline_matcher: types.ModuleType,
    experiment_source: str,
) -> dict[str, dict[str, str]]:
    selected = [row for row in rows if f"{row['seed']}/{row['step']}" in REGRESSION_IDENTITIES]
    selected_identities = {f"{row['seed']}/{row['step']}" for row in selected}
    if selected_identities != set(REGRESSION_IDENTITIES):
        raise ValueError("retained archive does not contain every rejected-experiment regression")

    full_matcher = _module_from_source("c26_rejected_experiment_full", experiment_source)
    no_promotion_matcher = _module_from_source(
        "c26_rejected_experiment_no_promotion", experiment_source
    )
    if not hasattr(no_promotion_matcher, "_promote_capped_tail_counterparts"):
        raise ValueError("rejected experiment omits the capped-source promotion hook")

    def no_promotion(observed, engine, *, support, target_side):
        return no_promotion_matcher._split_component_events(observed)

    no_promotion_matcher._promote_capped_tail_counterparts = no_promotion

    no_pre_support_matcher = _module_from_source(
        "c26_rejected_experiment_no_pre_support", experiment_source
    )
    if not hasattr(no_pre_support_matcher, "branch_event_legal_rolls"):
        raise ValueError("rejected experiment omits the pre-state support hook")

    def baseline_support(
        branch,
        *,
        side_one_choice,
        side_two_choice,
        pre_state=None,
    ):
        _ = pre_state
        return baseline_matcher.branch_event_legal_rolls(
            branch,
            side_one_choice=side_one_choice,
            side_two_choice=side_two_choice,
        )

    no_pre_support_matcher.branch_event_legal_rolls = baseline_support

    configurations = {
        "full_experiment": full_matcher,
        "without_capped_source_promotion": no_promotion_matcher,
        "without_pre_state_and_named_callee_support": no_pre_support_matcher,
    }
    actual: dict[str, dict[str, str]] = {}
    for name, matcher in configurations.items():
        _tally, results = _reread(selected, matcher)
        actual[name] = {
            identity: str(results[identity]["verdict"])
            for identity in REGRESSION_IDENTITIES
        }

    expected = {
        "full_experiment": {identity: "diverged" for identity in REGRESSION_IDENTITIES},
        "without_capped_source_promotion": {
            identity: "diverged" for identity in REGRESSION_IDENTITIES
        },
        "without_pre_state_and_named_callee_support": {
            identity: "matched" for identity in REGRESSION_IDENTITIES
        },
    }
    if actual != expected:
        raise ValueError("rejected-experiment regression ablation pattern changed")
    return actual


def verify(shard_glob: str) -> dict[str, Any]:
    from engine_build_fingerprint import assert_fresh, compute_fingerprint

    readout = _readout()
    baseline = readout["pinned_baseline"]
    control = readout["current_main_control"]
    experiment = readout["rejected_experiment"]
    equivalence = readout["final_main_equivalence"]
    commit = baseline["commit"]

    assert_fresh()
    fingerprint = compute_fingerprint()["fingerprint"]
    if fingerprint != baseline["engine_fingerprint"]:
        raise ValueError("engine fingerprint does not match the pinned C26 baseline")
    if control["commit"] != commit:
        raise ValueError("current-main control does not use the pinned baseline commit")
    if equivalence["matcher_path"] != MATCHER_PATH:
        raise ValueError("final-equivalence contract names the wrong matcher path")

    baseline_source = _source_at_commit(commit)
    if _source_sha256(baseline_source) != baseline["matcher_source_sha256"]:
        raise ValueError("pinned baseline matcher source SHA-256 does not match the evidence")
    final_source = (REPO_ROOT / MATCHER_PATH).read_text()
    if final_source != baseline_source:
        raise ValueError("final matcher source differs from the pinned baseline")
    provenance = equivalence["baseline_repro_provenance"]
    if provenance["classification"] != "current_main_baseline":
        raise ValueError("C27 repro provenance is not classified as baseline behavior")
    for field in provenance["fields"]:
        assignment = f'"{field}": prepared["{field}"]'
        if assignment not in baseline_source:
            raise ValueError(f"pinned baseline omits required repro provenance field {field}")

    rows = _verify_archive_inputs(readout, shard_glob)
    baseline_matcher = _module_from_source("c26_pinned_main_matcher", baseline_source)
    final_matcher = _module_from_source("c26_final_matcher", final_source)
    baseline_tally, baseline_rows = _reread(rows, baseline_matcher)
    final_tally, final_rows = _reread(rows, final_matcher)

    if baseline_tally != control["tally"]:
        raise ValueError("pinned-main reread tally differs from the C26 evidence")
    if final_tally != control["tally"]:
        raise ValueError("final reread tally differs from the C26 evidence")

    for expected in control["rows"]:
        actual = final_rows.get(expected["identity"])
        if actual != {"verdict": expected["verdict"], "branches": expected["branches"]}:
            raise ValueError("final C15 identity result differs from the C26 evidence")

    delta = _verdict_delta(baseline_rows, final_rows)
    expected_delta = equivalence["archive_reread_delta"]
    observed_delta = {
        "diverged_to_matched": delta.get("diverged_to_matched", 0),
        "matched_to_diverged": delta.get("matched_to_diverged", 0),
    }
    if observed_delta != expected_delta or len(delta) != 0:
        raise ValueError("final matcher has a nonzero retained-archive verdict delta")

    if experiment["matcher_path"] != MATCHER_PATH:
        raise ValueError("rejected experiment names the wrong matcher path")
    experiment_source = _source_at_commit(experiment["commit"])
    if _source_sha256(experiment_source) != experiment["matcher_source_sha256"]:
        raise ValueError("rejected experiment matcher SHA-256 does not match the evidence")
    regression_ablations = _regression_ablation_check(
        rows,
        baseline_matcher=baseline_matcher,
        experiment_source=experiment_source,
    )

    return {
        "schema": "c26-damage-composition-verifier/1",
        "status": "verified",
        "population": len(rows),
        "engine_fingerprint": fingerprint,
        "pinned_main_tally": baseline_tally,
        "final_tally": final_tally,
        "verdict_delta": observed_delta,
        "rejected_experiment_regression_ablations": regression_ablations,
        "baseline_repro_provenance_fields": provenance["fields"],
        "archive_shards": [shard["label"] for shard in baseline["retained_archive"]["shards"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--shards",
        required=True,
        help="glob matching the eight retained archive shards; required for execution",
    )
    parser.add_argument("--json", type=Path, help="write the public verification result")
    args = parser.parse_args(argv)

    payload = verify(args.shards)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.json is not None:
        args.json.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
