#!/usr/bin/env python3
"""Check that C26's retired renderer is covered by the clean confusion lane.

This is intentionally a source-and-fixture comparison: C26 retained only the
three public identity labels, not its replay payloads or checkpoint shard.
It therefore proves the committed renderer contract without manufacturing a
classifier result from unavailable inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


CLEAN_REV = "51faf308a8a3bb626b1c3d2e5b12b0491abaea5c"
RETIRED_REV = "3ec869c45e95169faa6d347ab8164a593e4ca097"
EVENTS = "rust/pokezero-search/src/events.rs"
NATIVE_TEST = "rust/pokezero-search/tests/gen3_confusion_event_renderer.rs"
PREDICTION = "reports/c26_switch_confusion_event_attribution_prediction.md"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("../pokezero-confusion-before-substitute-clean"),
        help="read-only worktree containing the clean confusion lane",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    prediction = (repo / PREDICTION).read_text(encoding="utf-8")
    mentioned_identities = tuple(re.findall(r"\b\d{7}/\d+\b", prediction))
    if set(mentioned_identities) != set(EXPECTED_IDENTITIES):
        raise RuntimeError(f"C26 identity ledger changed: {mentioned_identities!r}")
    identities = EXPECTED_IDENTITIES

    candidate = args.candidate.resolve()
    git(candidate, "cat-file", "-e", f"{CLEAN_REV}^{{commit}}")
    clean_events = git(candidate, "show", f"{CLEAN_REV}:{EVENTS}")
    clean_tests = git(candidate, "show", f"{CLEAN_REV}:{NATIVE_TEST}")
    retired_events = git(repo, "show", f"{RETIRED_REV}:{EVENTS}")

    # The retired lane recognized only a structural pair and emitted a tagged
    # line. That conflicts with the V2/V3 contract for the public stream.
    require(retired_events, "fn render_confusion_self_hit(", "retired pair matcher")
    require(retired_events, "|[from] confusion", "retired tagged output")

    # The replacement is independent of the preceding switch: switch handling
    # completes before this move phase. It consumes the real prelude, proves the
    # fixed 40-power damage identity, and rejects collapsed crash/self-faint
    # deltas rather than inventing a confusion event.
    require(clean_events, "fn confusion_self_hit_damage(", "damage derivation")
    require(clean_events, "fn classify_confusion_self_hit(", "fail-closed classifier")
    require(
        clean_events,
        "crash_matches || self_faint_move_can_be_self_only",
        "collision guard",
    )
    require(
        clean_events,
        "out.lines.push(format!(\"|-damage|{ident}|{condition}\"));",
        "untagged exact self-hit output",
    )
    require(clean_tests, "fn crash_miss_remains_a_move_not_a_confusion_self_hit()", "crash control")
    require(
        clean_tests,
        "fn explosion_behind_protect_is_not_misrendered_as_confusion()",
        "self-faint control",
    )

    evidence = {
        "verdict": "superseded_at_renderer_contract",
        "retired_commit": RETIRED_REV,
        "replacement_commit": CLEAN_REV,
        "identities": [
            {
                "identity": identity,
                "documented_shape": "voluntary switch, then opposing exact confusion self-hit",
                "replacement_path": "post-switch render_move_phase -> exact confusion classifier",
            }
            for identity in identities
        ],
        "v2_v3_contract": "replacement emits untagged self-hit damage; V2 remains folded and V3 corrects its own damage column",
        "row_replay": "not asserted: no C26 row payload, checkpoint shard, or result artifact is retained in this repository",
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
