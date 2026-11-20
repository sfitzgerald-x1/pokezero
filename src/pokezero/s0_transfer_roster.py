"""Finite inputs and primitive cells for the S0 transfer assays."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA = "pokezero.s0-transfer-cells.r2"
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


def write_once(path, value):
    """Publish an immutable artifact atomically, even with competing producers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(value) + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        try:
            temp.write(content)
            temp.flush()
            os.fsync(temp.fileno())
            try:
                os.link(temp.name, path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise ValueError(f"immutable artifact changed: {path}") from None
        finally:
            os.unlink(temp.name)
    return hashlib.sha256(content).hexdigest()


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


def actor_bindings(manifest, cell):
    pair = cell["role"], cell["opponent"]
    if cell["seat"] == 1:
        pair = pair[::-1]
    return [
        dict(role=role, **({name: manifest["actors"][cell["replica"]][role][name]
                           for name in ("checkpoint_sha256", "archive_sha256", "model_sha256", "self_moves")}
                           if role != "FP" else {"revision": manifest["fp_revision"], "budget_ms": manifest["fp_budget_ms"]}))
        for role in pair
    ]


def journal_path(root, cell):
    return root / f'{cell["environment"]}.journal' / f"{digest(cell_key(cell))}.json"


def cell_record(manifest, cell, status, score, evidence):
    start = manifest["starts"][str(cell["cluster"])]
    return dict(cell, manifest_sha256=digest(manifest), start_sha256=digest(start),
                actor_bindings=actor_bindings(manifest, cell), status=status, score=score,
                evidence=evidence, evidence_sha256=digest(evidence))
