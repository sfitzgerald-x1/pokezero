"""Finite inputs and primitive cells for the S0 transfer assays."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA = "pokezero.s0-transfer.r2-simple"
DRY_RUN_ROOT = 2_080_000_000
DRY_RUN_PAIRS = 20
H2H = "matched_h2h_sampled"
FP = "external_fp_greedy"
DOMAINS = (
    ("engine", H2H, "chance"), ("engine", H2H, "action"),
    ("showdown", H2H, "chance"), ("showdown", H2H, "action"),
    ("showdown", FP, "chance"), ("showdown", FP, "action"),
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value):
    """Publish a whole JSON file for another process to read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        try:
            temp.write(canonical(value) + b"\n")
            temp.close()
            os.replace(temp.name, path)
        finally:
            Path(temp.name).unlink(missing_ok=True)


def address(environment, assay, kind, cluster):
    domain = DOMAINS.index((environment, assay, kind))
    # Use a uint32 bijection after partitioning the registered seed bands.
    return ((cluster + domain * 300_000_000) * 2_654_435_761) % 2**32


def dry_run_clusters():
    """Dedicated scratch plumbing band, disjoint from the declared S0 bands."""
    return {
        H2H: list(range(DRY_RUN_ROOT, DRY_RUN_ROOT + DRY_RUN_PAIRS)),
        FP: list(range(DRY_RUN_ROOT + 1000, DRY_RUN_ROOT + 1000 + DRY_RUN_PAIRS)),
    }


def expected_cells(banks, *, dry_run):
    """Resolve cell identity before play. Shard order cannot re-key a cell."""
    expected = {}
    seen = set()
    for replica, bank in sorted(banks.items()):
        if set(bank) != {H2H, FP}:
            raise ValueError("both assay banks are required")
        if dry_run and bank != dry_run_clusters():
            raise ValueError("dry run must use the dedicated scratch band")
        for assay, clusters in sorted(bank.items()):
            if not clusters or any(type(c) is not int or not 0 < c < 2**31 for c in clusters):
                raise ValueError("nonempty positive int32 cluster bank required")
            if not dry_run and any(DRY_RUN_ROOT <= c < DRY_RUN_ROOT + 10_000 for c in clusters):
                raise ValueError("scored banks must be disjoint from scratch and warmup")
            if len(set(clusters)) != len(clusters) or seen.intersection(clusters):
                raise ValueError("cluster banks overlap")
            seen.update(clusters)
            if assay == H2H:
                games = tuple((env, role, "B") for env in ("engine", "showdown") for role in ("T", "C"))
            else:
                roles = ("T", "B", "C") if dry_run else ("T", "B", "C", "H")
                games = tuple(("showdown", role, "FP") for role in roles)
            for cluster in clusters:
                for environment, role, opponent in games:
                    for seat in (0, 1):
                        key = (replica, assay, cluster, environment, role, seat)
                        expected[key] = dict(
                            replica=replica, assay=assay, cluster=cluster,
                            environment=environment, role=role, seat=seat, opponent=opponent,
                            chance=address(environment, assay, "chance", cluster),
                            action=address(environment, assay, "action", cluster),
                        )
    unique_addresses = {
        (cell["assay"], cell["cluster"], cell["environment"], kind): cell[kind]
        for cell in expected.values() for kind in ("chance", "action")
    }
    if len(set(unique_addresses.values())) != len(unique_addresses):
        raise ValueError("derived RNG addresses collide")
    return expected


CELL_KEYS = ("replica", "assay", "cluster", "environment", "role", "seat")


def cell_key(cell):
    return tuple(cell[key] for key in CELL_KEYS)


def read_terminals(path, *, manifest_sha256):
    """Read complete JSONL records, retaining outcomes from an interrupted worker."""
    if not path.is_file():
        return
    with path.open("rb") as stream:
        header = stream.readline()
        if not header.endswith(b"\n"):
            return
        if json.loads(header) != {"manifest_sha256": manifest_sha256}:
            raise ValueError(f"{path.name} belongs to a different run manifest")
        for line in stream:
            if not line.endswith(b"\n"):
                # Process termination can leave the last record partially written.
                break
            yield json.loads(line)
