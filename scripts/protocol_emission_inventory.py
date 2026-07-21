"""Build the v3 silent-noop sweep's static E/O/C protocol differential."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokezero.protocol_emission_inventory import build_protocol_inventory  # noqa: E402
from pokezero.audit_provenance import public_repo_commit  # noqa: E402
from pokezero.randbat import load_gen3_randbat_source_cached  # noqa: E402
from pokezero.showdown import observation_schema_version_from_choice  # noqa: E402


def _write_json_atomic(path: Path, payload: dict) -> None:
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
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--observed-audit", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--observation-schema",
        choices=("v3",),
        required=True,
        help="The silent-noop sweep is v3-only; schema fallback is forbidden.",
    )
    args = parser.parse_args(command_arguments)
    observation_schema = observation_schema_version_from_choice(args.observation_schema)
    if observation_schema is None:  # Defensive: argparse only accepts v3.
        raise AssertionError("protocol inventory requires an explicit observation schema")

    source = load_gen3_randbat_source_cached(args.showdown_root)
    payload = build_protocol_inventory(
        showdown_root=args.showdown_root,
        public_root=ROOT,
        observed_audits=args.observed_audit,
    )
    for entry in payload["observed"]["audit_provenance"]:
        provenance = entry["audit_provenance"]
        if provenance.get("observation_schema") != observation_schema:
            parser.error(f"observed audit has a non-v3 schema: {entry['path']}")
        if provenance.get("showdown_source_hash") != source.metadata.source_hash:
            parser.error(f"observed audit source hash differs from --showdown-root: {entry['path']}")
    payload["audit_provenance"] = {
        "schema_version": "pokezero.protocol-emission-inventory-provenance.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "public_repo_commit": public_repo_commit(ROOT),
        "showdown_source_hash": source.metadata.source_hash,
        "observation_schema": observation_schema,
        "image_digest": os.environ.get("POKEZERO_AUDIT_IMAGE_DIGEST", "local-uncontainerized"),
        "command": [str(Path(__file__).relative_to(ROOT)), *command_arguments],
    }
    _write_json_atomic(args.out, payload)
    differential = payload["differential"]
    print(
        "protocol emission inventory: "
        f"E={payload['engine_emittable']['tag_count']} "
        f"O={payload['observed']['tag_count']} "
        f"C={payload['consumer_dispatch']['tag_count']} "
        f"O-C(tags)={len(differential['observed_but_unconsumed'])} "
        f"O-C(signatures)={len(differential['observed_signatures_without_direct_consumer'])} "
        f"unclassified={len(differential['observed_signatures_without_semantic_coverage'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
