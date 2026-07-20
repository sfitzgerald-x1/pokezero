"""Audit public decision captures for distinct states with identical model inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokezero.encoding_collision_audit import audit_collision_sketches, audit_public_decision_corpus  # noqa: E402
from pokezero.audit_provenance import public_repo_commit  # noqa: E402
from pokezero.local_showdown import LocalShowdownConfig  # noqa: E402
from pokezero.randbat import load_gen3_randbat_source_cached  # noqa: E402
from pokezero.showdown import observation_schema_version_from_choice  # noqa: E402


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main(argv: Iterable[str] | None = None) -> int:
    command_arguments = list(argv) if argv is not None else list(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--corpus", type=Path, help="Full public-decision corpus to audit.")
    inputs.add_argument(
        "--collision-sketch",
        type=Path,
        action="append",
        dest="collision_sketches",
        help="Compact public-only sketch shard to audit; repeat for multiple shards.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--showdown-root", type=Path, default=None)
    parser.add_argument("--max-decisions", type=int, default=100_000)
    parser.add_argument("--start-decision", type=int, default=0)
    parser.add_argument(
        "--observation-schema",
        choices=("v3",),
        required=True,
        help="Audit v3 inputs only; default-schema fallback is forbidden.",
    )
    args = parser.parse_args(command_arguments)
    if args.max_decisions <= 0:
        parser.error("--max-decisions must be positive")
    if args.start_decision < 0:
        parser.error("--start-decision must be non-negative")
    if args.corpus is not None and not args.corpus.is_file():
        parser.error("--corpus must name an existing public-decision corpus")
    if args.collision_sketches is not None and any(not path.is_file() for path in args.collision_sketches):
        parser.error("each --collision-sketch must name an existing compact sketch")

    observation_schema = observation_schema_version_from_choice(args.observation_schema)
    if observation_schema is None:  # Defensive: the CLI requires an explicit v3 choice.
        raise AssertionError("collision audit requires an explicit observation schema")
    config = LocalShowdownConfig(showdown_root=args.showdown_root)
    source = load_gen3_randbat_source_cached(config.resolved_showdown_root())
    if args.corpus is not None:
        payload = audit_public_decision_corpus(
            args.corpus,
            max_decisions=args.max_decisions,
            start_decision=args.start_decision,
            expected_observation_schema=observation_schema,
        )
    else:
        assert args.collision_sketches is not None
        payload = audit_collision_sketches(
            args.collision_sketches,
            max_decisions=args.max_decisions,
            start_decision=args.start_decision,
            expected_observation_schema=observation_schema,
        )
    payload["audit_provenance"] = {
        "schema_version": "pokezero.encoding-collision-provenance.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "public_repo_commit": public_repo_commit(ROOT),
        "showdown_source_hash": source.metadata.source_hash,
        "observation_schema": observation_schema,
        "image_digest": os.environ.get("POKEZERO_AUDIT_IMAGE_DIGEST", "local-uncontainerized"),
        "command": [str(Path(__file__).relative_to(ROOT)), *command_arguments],
    }
    _write_json_atomic(args.out, payload)
    print(
        "encoding collision audit: "
        f"records={payload['records_scanned']} collisions={payload['collision_group_count']} "
        f"actionable={payload['actionable_collision_group_count']}"
    )
    # A collision is evidence, not a runtime failure. The persistent job uses
    # the artifact to decide whether to send a finding alert and always writes
    # its terminal marker after a valid audit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
