#!/usr/bin/env python3
"""Check that C26 is superseded by the public confusion implementation.

This is intentionally an immutable source-and-fixture comparison. C26 retained
only the three public identity labels, not its replay payloads or checkpoint
shard. It therefore proves the public merge's renderer contract without
manufacturing a classifier result from unavailable inputs.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


PUBLIC_MERGE = "8af4f42e99ef9b6a0b809027976a27a8d135cd3c"
EVENTS = "rust/pokezero-search/src/events.rs"
NATIVE_TEST = "rust/pokezero-search/tests/gen3_confusion_event_renderer.rs"
PREDICTION = "reports/c26_switch_confusion_event_attribution_prediction.md"
SUPERSESSION = "reports/c26_switch_confusion_event_attribution_supersession.md"
EXPECTED_IDENTITIES = ("3001000/57", "3300017/60", "3300122/21")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def require(source: str, fragment: str, label: str) -> None:
    if fragment not in source:
        raise RuntimeError(f"missing {label}: {fragment!r}")


def forbid(source: str, fragment: str, label: str) -> None:
    if fragment in source:
        raise RuntimeError(f"unexpected {label}: {fragment!r}")


def identities_in(source: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\b\d{7}/\d+\b", source))


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    prediction = (repo / PREDICTION).read_text(encoding="utf-8")
    supersession = (repo / SUPERSESSION).read_text(encoding="utf-8")
    for label, document in (("prediction", prediction), ("supersession", supersession)):
        mentioned_identities = identities_in(document)
        if set(mentioned_identities) != set(EXPECTED_IDENTITIES):
            raise RuntimeError(f"C26 {label} ledger changed: {mentioned_identities!r}")

    # Read the immutable, publicly merged implementation rather than a moving
    # branch or a feature-lane hash.
    git(repo, "cat-file", "-e", f"{PUBLIC_MERGE}^{{commit}}")
    parents = git(repo, "show", "-s", "--format=%P", PUBLIC_MERGE).split()
    if len(parents) != 2:
        raise RuntimeError(f"public implementation is not a merge commit: {parents!r}")
    public_events = git(repo, "show", f"{PUBLIC_MERGE}:{EVENTS}")
    public_tests = git(repo, "show", f"{PUBLIC_MERGE}:{NATIVE_TEST}")

    # Switch handling completes before the common move phase. The merged
    # classifier consumes the real prelude, proves the fixed 40-power identity,
    # emits canonical untagged self-hit damage, and fails closed on collisions.
    require(public_events, "fn confusion_self_hit_damage(", "damage derivation")
    require(public_events, "fn classify_confusion_self_hit(", "fail-closed classifier")
    require(
        public_events,
        "crash_matches || self_faint_move_can_be_self_only",
        "collision guard",
    )
    require(
        public_events,
        'out.lines.push(format!("|-activate|{ident}|confusion"));',
        "confusion activation",
    )
    require(
        public_events,
        "out.lines.push(format!(\"|-damage|{ident}|{condition}\"));",
        "untagged exact self-hit output",
    )
    forbid(public_events, "|[from] confusion", "tagged confusion output")
    require(public_tests, "fn exact_self_hit_renders_activation_and_cancels_substitute()", "exact self-hit test")
    require(public_tests, "fn crash_miss_remains_a_move_not_a_confusion_self_hit()", "crash control")
    require(
        public_tests,
        "fn explosion_behind_protect_is_not_misrendered_as_confusion()",
        "self-faint control",
    )
    require(
        public_tests,
        "fn recoil_after_an_executed_move_is_not_confusion_damage()",
        "ordinary-damage negative control",
    )
    require(public_tests, "[from] Recoil", "ordinary-damage source assertion")

    evidence = {
        "verdict": "superseded_by_public_merge_renderer_contract",
        "public_merge": PUBLIC_MERGE,
        "public_merge_parents": parents,
        "identities": [
            {
                "identity": identity,
                "documented_shape": "voluntary switch, then opposing exact confusion self-hit",
                "replacement_path": "post-switch render_move_phase -> exact confusion classifier",
            }
            for identity in EXPECTED_IDENTITIES
        ],
        "v2_v3_contract": "public merge emits untagged self-hit damage; V2 remains folded and V3 corrects its own damage column",
        "ordinary_damage_control": "merged native Recoil test preserves [from] Recoil rather than confusion attribution",
        "row_replay": "not asserted: no C26 row payload, checkpoint shard, or result artifact is retained in this repository",
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
