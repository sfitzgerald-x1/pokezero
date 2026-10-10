"""Bounded authored-only public-view component timings, never qualification.

Uses test fixtures and FakeSetSource, not real/exposed roots, model forwards,
native search or historical replays. Writes a fresh contract before measuring.
No attempt to recover the failed real atomic probes or infer playing strength.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from pokezero.policy_opponent_diagnostics import PolicyOpponentDiagnostics, VIEW_PHASES
from pokezero.policy_opponent_view import (
    _PublicPolicyPrefix, build_policy_opponent_view_from_native_bundle, public_policy_lines,
)
from test_policy_opponent_request import arguments, bundle
from test_policy_opponent_view import LINES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def authored_lines():
    lines = list(LINES)
    for turn in range(1, 71):
        lines.extend(("|move|p1a: Swampert|Surf|p2a: Snorlax",
            "|-damage|p2a: Snorlax|300/400",
            "|move|p2a: Snorlax|Body Slam|p1a: Swampert",
            "|-damage|p1a: Swampert|150/300",
            "|-heal|p1a: Swampert|201/300|[from] item: Leftovers",
            "|-heal|p2a: Snorlax|400/400|[from] item: Leftovers",
            "|upkeep", f"|turn|{turn + 1}"))
    return public_policy_lines(lines, hp_visibility={"p1": "exact", "p2": "exact"})


def assert_parity(reference, actual):
    assert reference.state == actual.state
    assert reference.public_lines == actual.public_lines
    assert reference.native_action_indices == actual.native_action_indices
    assert reference.materialization.replay == actual.materialization.replay
    assert reference.materialization.self_request == actual.materialization.self_request
    assert reference.materialization.self_move_states == actual.materialization.self_move_states
    assert vars(reference.materialization.belief_engine) == vars(actual.materialization.belief_engine)


def run(output):
    # This bounded diagnostic intentionally uses assertions as independent
    # parity witnesses. Never emit a readout when the interpreter removes them.
    if sys.flags.optimize:
        raise ValueError("authored component profiling requires enabled assertions; optimization is refused")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("fresh authored profile directory required; no overwrite")
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in (
        Path(__file__).resolve(), ROOT / "src/pokezero/policy_opponent_view.py",
        ROOT / "src/pokezero/policy_opponent_diagnostics.py", ROOT / "src/pokezero/showdown.py",
        ROOT / "src/pokezero/belief.py", ROOT / "tests/test_policy_opponent_request.py",
        ROOT / "tests/test_policy_opponent_view.py", ROOT / "tests/test_showdown.py")}
    root = authored_lines()
    lines = (*root, "|move|p1a: Swampert|Surf|p2a: Snorlax", "|-damage|p2a: Snorlax|75/100")
    kwargs = arguments(lines)
    kwargs["hp_visibility"] = {"p1": "percentage", "p2": "percentage"}
    supplied = bundle()
    before = deepcopy(supplied)
    schedule = (("full_rebuild", 10), ("cold_cached", 10), ("warm_cached", 40))
    contract = dict(schema="pokezero.public-view.authored-components-contract.v1",
        status="AUTHORED_ONLY_NOT_SCIENTIFIC_OR_RUNTIME_QUALIFICATION", input_sha256=inputs,
        root_lines=len(root), root_characters=sum(map(len, root)), suffix_lines=2,
        root_sha256=hashlib.sha256(json.dumps(root).encode()).hexdigest(),
        schedule=[dict(mode=mode, repetitions=n) for mode, n in schedule],
        set_source="FakeSetSource; not production set-source throughput",
        phases=list(VIEW_PHASES), outputs="aggregate numerical timings and parity only",
        warmup="One excluded warm_cached construction before timed warm invocations",
        scope=dict(real_roots=0, historical_replays=0, model_forwards=0, native_search_calls=0,
            scientific_outcomes=0, causal_runtime_claim=False, admission=False))
    output.mkdir(parents=True)
    save_new(output / "contract.json", contract)
    reference = build_policy_opponent_view_from_native_bundle(native_request_bundle=supplied, **kwargs)
    profiles = []
    for mode, n in schedule:
        sink = PolicyOpponentDiagnostics()
        prefix = _PublicPolicyPrefix(root, battle_id=kwargs["battle_id"]) if mode == "warm_cached" else None
        if prefix is not None:
            assert_parity(reference, build_policy_opponent_view_from_native_bundle(
                native_request_bundle=supplied, **kwargs, _public_prefix=prefix))
        root_before = deepcopy(vars(prefix._parser)) if prefix is not None else None
        for _ in range(n):
            if mode == "cold_cached":
                prefix = _PublicPolicyPrefix(root, battle_id=kwargs["battle_id"])
            with sink.phase("view_reconstruction"):
                actual = build_policy_opponent_view_from_native_bundle(native_request_bundle=supplied,
                    **kwargs, _public_prefix=prefix, diagnostics=sink)
            assert_parity(reference, actual)
            assert supplied == before
        if mode == "warm_cached":
            assert root_before == vars(prefix._parser)
        snapshot = sink.snapshot()
        assert snapshot["valid"] and snapshot["timing_faults"] == 0
        rows = snapshot["phases"]
        assert rows["view_reconstruction"]["calls"] == n
        assert all(row["timed_calls"] == row["calls"] and row["failed_calls"] == 0 for row in rows.values())
        assert rows["view_prefix_prepare"]["calls"] == (n if mode == "cold_cached" else 0)
        for phase in ("view_branch_clone", "view_suffix_parse", "view_replay_snapshot"):
            assert rows[phase]["calls"] == (0 if mode == "full_rebuild" else n)
        total = rows["view_reconstruction"]["elapsed_seconds"]
        children = ("view_public_projection", "view_own_request_certification", "view_replay_parse",
            "view_belief_rebuild", "view_normalization", "view_materialization")
        residual = total - sum(rows[p]["elapsed_seconds"] for p in children)
        assert residual >= -1e-12
        profiles.append(dict(mode=mode, repetitions=n, snapshot=snapshot,
            inclusive_view_mean_seconds=total/n,
            direct_component_fractions={p: rows[p]["elapsed_seconds"]/total for p in children},
            unsegmented_view_overhead_seconds=max(0., residual),
            parity=dict(state=True, public_replay=True, full_beliefs=True, own_request_and_pp=True,
                native_action_map=True, input_unmutated=True, warm_prefix_unmutated=mode == "warm_cached")))
    assert all(sha(ROOT / path) == digest for path, digest in inputs.items()), "profile source drift"
    result = dict(schema="pokezero.public-view.authored-components-readout.v1",
        status="COMPLETE_AUTHORED_COMPONENT_PROFILE_ONLY", contract_sha256=sha(output / "contract.json"),
        profiles=profiles, scope=contract["scope"], limitations=[
            "Authored 70-turn transcript and two-Pokemon request with FakeSetSource, not a representative battle.",
            "No model/native search, deadlines, production checkpoint or production belief source was exercised.",
            "Inclusive replay contains prepare/clone/suffix/snapshot; do not double-count child spans.",
            "Group ordering and instrumentation prevent interpreting modes as a causal production speedup.",
            "Component shares guide further investigation only; no scientific or uninstrumented runtime qualification."])
    save_new(output / "readout.json", result)
    print(json.dumps(dict(status=result["status"], output=str(output),
        warm_components=profiles[-1]["direct_component_fractions"])), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output))
