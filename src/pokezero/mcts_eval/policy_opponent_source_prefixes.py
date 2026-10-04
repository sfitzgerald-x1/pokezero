"""Source-byte/prefix preflight, deliberately not live replay qualification.

No model or search is invoked. Every frozen root keeps its place, including
source-owned prefix refusals. Integrity failures stop the whole preflight;
eligibility refusals never authorize a replacement root.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .policy_opponent_roster import RosterError, load_frozen_roster, verify_source_files
from .source_root_replay import SourceRootReplayError, source_bound_replay_prefix
from ..public_decision_corpus import PublicDecisionRecord


SCHEMA = "pokezero.paper-policy-opponent-source-prefix-preflight.v1"


def qualify_source_prefixes(
    roster_path: str | Path, *, expected_roster_sha256: str, source_root: str | Path,
) -> dict[str, Any]:
    roster = load_frozen_roster(roster_path, expected_sha256=expected_roster_sha256)
    base = Path(source_root).resolve(strict=True)
    verify_source_files(base, roster)
    references = {row["source_relative_path"]: row
                  for row in roster["profile_roots"] + roster["prefix_witnesses"]}
    records = {}
    for relative, row in references.items():
        path = (base / relative).resolve(strict=True)
        if not path.is_relative_to(base):
            raise RosterError("source path escapes cohort during preflight")
        raw = path.read_bytes()
        # Verify the bytes actually decoded, not only an earlier read of them.
        if hashlib.sha256(raw).hexdigest() != row["source_file_sha256"]:
            raise RosterError(f"source bytes drift during preflight: {row['decision_id']}")
        try:
            records[relative] = PublicDecisionRecord.from_dict(json.loads(raw)["record"])
        except (ValueError, TypeError, KeyError) as error:
            raise RosterError(f"invalid canonical source record: {row['decision_id']}") from error
    results = []
    for row in roster["profile_roots"]:
        record = records[row["source_relative_path"]]
        result = dict(row)
        try:
            prefix = source_bound_replay_prefix(record, source_records=tuple(records.values()))
            result.update(state="PREFIX_VALID", public_rounds=len(prefix.public_action_rounds),
                          source_owned_repairs=[repair.to_dict() for repair in prefix.repairs])
        except SourceRootReplayError as error:
            result.update(state="REFUSED", refusal=str(error), source_owned_repairs=[])
        results.append(result)
    # A source/selection change during this read-only check invalidates it.
    load_frozen_roster(roster_path, expected_sha256=expected_roster_sha256)
    verify_source_files(base, roster)
    terminal = f"shards/{roster['source_shard']}/COMPLETE.json"
    inventory = [{"source_relative_path": relative, "source_file_sha256": row["source_file_sha256"]}
                 for relative, row in sorted(references.items())]
    inventory.append({"source_relative_path": terminal, "source_file_sha256": roster["source_terminal_sha256"]})
    return {
        "schema_version": SCHEMA, "status": "SOURCE_PREFIX_PREFLIGHT_COMPLETE",
        "roster_sha256": expected_roster_sha256, "source_cohort": roster["source_cohort"],
        "root_denominator": len(results), "roots": results,
        "source_inventory": inventory, "source_files_verified": len(inventory),
        "prefix_valid": sum(row["state"] == "PREFIX_VALID" for row in results),
        "refused": sum(row["state"] == "REFUSED" for row in results),
        "live_replay_qualified": False, "profile_qualified": False,
        "replacement_roots": [],
    }
