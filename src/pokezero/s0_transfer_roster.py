"""Finite inputs and primitive cells for the S0 transfer assays."""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
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


@dataclass(frozen=True)
class Shard:
    index: int = 0
    count: int = 1

    @classmethod
    def parse(cls, text):
        index, count = map(int, text.split("/"))
        if not 0 <= index < count:
            raise ValueError("shard must be a zero-based I/N")
        return cls(index, count)

    def owns(self, cell):
        return self.count == 1 or int(digest(cell_key(cell))[:16], 16) % self.count == self.index

    def stem(self, side):
        return side if self.count == 1 else f"{side}-{self.index}-of-{self.count}"


def bind_shards(root, side, count, manifest_sha):
    # Serialized once per worker, so competing shard counts cannot both start.
    with (root / f"{side}.layout.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / f"{side}.layout.json"
        layout = dict(count=count, manifest_sha256=manifest_sha)
        if path.exists():
            if json.loads(path.read_bytes()) != layout:
                raise ValueError("shard layout or manifest changed")
        else:
            if count != 1 and (root / f"{side}.jsonl").exists():
                raise ValueError("existing unsharded ledger")
            write_json(path, layout)


def raw_paths(root, side, manifest_sha):
    layout = root / f"{side}.layout.json"
    if not layout.exists():  # original unsharded ledgers
        return [root / f"{side}.jsonl"]
    binding = json.loads(layout.read_bytes())
    count = binding["count"]
    if binding["manifest_sha256"] != manifest_sha or type(count) is not int or count < 1:
        raise ValueError("invalid shard layout")
    return [root / (Shard(i, count).stem(side) + ".jsonl") for i in range(count)]


def read_terminals(path, *, manifest_sha256, hasher=None):
    """A partial trailing write is uncommitted. A malformed complete record is counted."""
    if not path.is_file():
        return
    with path.open("rb") as stream:
        header = stream.readline()
        if hasher is not None:
            hasher.update(header)
        if not header.endswith(b"\n"):
            if header:
                raise ValueError("incomplete transfer manifest header")
            return
        if json.loads(header) != {"manifest_sha256": manifest_sha256}:
            raise ValueError(f"{path.name} belongs to a different run manifest")
        for line in stream:
            if hasher is not None:
                hasher.update(line)
            if not line.endswith(b"\n"):
                break
            try:
                yield json.loads(line)
            except (UnicodeError, json.JSONDecodeError):
                yield None


@contextmanager
def terminal_writer(root, side, shard, expected, manifest_sha, *, resume):
    bind_shards(root, side, shard.count, manifest_sha)
    path = root / (shard.stem(side) + ".jsonl")
    with path.open("a+b" if resume else "x+b") as output:
        fcntl.flock(output, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.seek(0)
        header = canonical(dict(manifest_sha256=manifest_sha)) + b"\n"
        line = output.readline()
        if line != header:
            if not header.startswith(line):
                raise ValueError("ledger belongs to a different run manifest")
            output.seek(0)
            output.truncate()
            output.write(header)
            output.flush()
        remaining = dict(expected)
        for row in iter(output.readline, b""):
            if not row.endswith(b"\n"):
                output.truncate(output.tell() - len(row))
                break
            try:
                key = cell_key(json.loads(row))
                remaining.pop(key, None)  # includes refused and failed attempts
            except (ValueError, TypeError, KeyError, UnicodeError):
                # Unattributable attempted cell: never replay unresolved identities.
                raise ValueError("cannot resume a shard with an unidentifiable committed cell")
        def publish(result):
            cell, status, witness = result
            if remaining.pop(cell_key(cell), None) != cell:
                raise ValueError("unexpected or duplicate terminal in shard")
            output.write(canonical(dict(cell, status=status, evidence=witness)) + b"\n")
            output.flush()
        yield tuple(remaining.values()), publish
