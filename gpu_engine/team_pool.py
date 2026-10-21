"""Pack a one-time real Showdown team pool; no battle or network inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


FIELDS = {
    "species": ((6,), np.uint16), "level": ((6,), np.uint8),
    "types": ((6, 2), np.uint8), "stats": ((6, 6), np.int32),
    "moves": ((6, 4), np.uint16), "pp": ((6, 4), np.int16),
    "item": ((6,), np.uint16), "ability": ((6,), np.uint16),
    "nature": ((6,), np.uint8), "ivs": ((6, 6), np.uint8),
    "evs": ((6, 6), np.uint16), "hidden_power_type": ((6,), np.uint8),
    "hidden_power_power": ((6,), np.uint8),
}


def pack_stream(stream, *, count, seed_start):
    header = json.loads(next(stream))
    if (header.get("kind") != "header" or header.get("schema") != "pokezero.gpu-team-stream.v1"
            or header.get("team_count") != count or header.get("seed_start") != seed_start):
        raise ValueError("unexpected team stream header")
    arrays = {key: np.empty((count, *shape), dtype=dtype) for key, (shape, dtype) in FIELDS.items()}
    arrays["seeds"] = np.arange(seed_start, seed_start + count, dtype=np.uint32)
    for i in range(count):
        row = json.loads(next(stream))
        if row.get("kind") != "team" or row.get("index") != i or row.get("seed") != seed_start + i:
            raise ValueError("team stream is not the fixed chronological prefix")
        members = row["members"]
        if len(members) != 6:
            raise ValueError("six members required")
        for key, (shape, dtype) in FIELDS.items():
            raw = [m[key] for m in members]
            # NumPy promotes mixed bool/int rows to integers; reject before coercion.
            def integral_tree(value):
                return (all(integral_tree(v) for v in value) if isinstance(value, list)
                        else type(value) is int)
            if not integral_tree(raw):
                raise ValueError(f"noninteger input for {key}")
            value = np.asarray(raw)
            limits = np.iinfo(dtype)
            if (value.shape != shape or value.dtype.kind not in "iu"
                    or np.any(value < limits.min) or np.any(value > limits.max)):
                raise ValueError(f"invalid shape/range/type for {key}")
            arrays[key][i] = value
        if (np.any(arrays["stats"][i] <= 0)
                or np.any(arrays["level"][i] < 1) or np.any(arrays["level"][i] > 100)):
            raise ValueError("nonpositive stats/PP or invalid level")
        for field, vocabulary in [("species", "species"), ("moves", "moves"),
                                  ("types", "types"), ("item", "items"),
                                  ("ability", "abilities"), ("nature", "natures"),
                                  ("hidden_power_type", "types")]:
            if np.any(arrays[field][i] >= len(header["vocabulary"][vocabulary])):
                raise ValueError(f"out-of-vocabulary {field}")
        for field in ("species", "ability", "nature", "hidden_power_type"):
            if np.any(arrays[field][i] == 0):
                raise ValueError(f"missing required {field}")
        missing_move = arrays["moves"][i] == 0
        if (np.any(np.all(missing_move, axis=-1))
                or np.any((arrays["pp"][i] == 0) != missing_move)
                or np.any(arrays["pp"][i] < 0)):
            raise ValueError("each member needs a move; only empty move slots have zero PP")
        if np.any(missing_move[:, :-1] & ~missing_move[:, 1:]):
            raise ValueError("empty move slots must be trailing padding")
        if np.any(arrays["types"][i, :, 0] == 0):
            raise ValueError("missing primary type")
        if (np.any(arrays["hidden_power_power"][i] < 30)
                or np.any(arrays["hidden_power_power"][i] > 70)):
            raise ValueError("invalid Gen 3 Hidden Power power")
        hp_types = {"fighting", "flying", "poison", "ground", "rock", "bug", "ghost",
                    "steel", "fire", "water", "grass", "electric", "psychic", "ice",
                    "dragon", "dark"}
        if any(header["vocabulary"]["types"][int(t)] not in hp_types
               for t in arrays["hidden_power_type"][i]):
            raise ValueError("invalid Gen 3 Hidden Power type")
        if np.any(arrays["ivs"][i] > 31) or np.any(arrays["evs"][i] > 255):
            raise ValueError("invalid IV/EV range")
        if np.any(np.sum(arrays["evs"][i], axis=-1) > 510):
            raise ValueError("EV sum exceeds Gen 3 limit")
        if (i + 1) % 10000 == 0:
            print(f"POOL_TEAM_ROWS={i+1}/{count}", file=sys.stderr, flush=True)
    terminal = json.loads(next(stream))
    if terminal != {"kind": "terminal", "team_count": count}:
        raise ValueError("missing clean terminal team count")
    if next(stream, None) is not None:
        raise ValueError("unexpected trailing team output")
    values, counts = np.unique(arrays["moves"][arrays["moves"] != 0], return_counts=True)
    ranked = sorted(zip(values.tolist(), counts.tolist()), key=lambda x: (-x[1], x[0]))
    total = int(sum(counts))
    cumulative = 0
    ledger = []
    for move, n in ranked:
        cumulative += n
        ledger.append(dict(id=header["vocabulary"]["moves"][move], count=n,
                           share=n / total, cumulative_share=cumulative / total,
                           implemented=False))
    target = next(i + 1 for i, r in enumerate(ledger) if r["cumulative_share"] >= 0.85)
    header.update(schema="pokezero.gpu-team-pool.v1", completed_teams=count,
                  move_slots=int(total), move_frequency=ledger, target85_move_count=target,
                  tensor_move_slots=count * 24, empty_move_id=0,
                  frequency_basis="Actual generated move slots, not assumed catalog weights",
                  implemented_move_coverage=0.0, transition_implemented=False)
    arrays["metadata_utf8"] = np.frombuffer(json.dumps(header, sort_keys=True).encode(), dtype=np.uint8)
    return arrays, header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showdown-root", required=True, type=Path)
    parser.add_argument("--source-commit-label", required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--teams", type=int, default=200000)
    parser.add_argument("--seed-start", type=int, default=650000000)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite an existing team pool")
    if args.teams <= 0 or not 0 <= args.seed_start < args.seed_start + args.teams <= 2**32:
        raise ValueError("positive count and unsigned 32-bit seed range required")
    process = subprocess.Popen([args.node, str(Path(__file__).with_name("generate_teams.cjs")),
        str(args.showdown_root), str(args.teams), str(args.seed_start), args.source_commit_label],
        stdout=subprocess.PIPE, text=True)
    try:
        arrays, header = pack_stream(process.stdout, count=args.teams, seed_start=args.seed_start)
        code = process.wait(timeout=30)
        if code != 0:
            raise RuntimeError(f"Showdown team generator exited {code}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic and exclusive publication: never replace an existing dataset.
    with tempfile.NamedTemporaryFile(dir=args.out.parent, suffix=".partial", delete=False) as output:
        temporary = Path(output.name)
        try:
            np.savez_compressed(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
            os.link(temporary, args.out)
        finally:
            temporary.unlink(missing_ok=True)
    with args.out.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps(dict(status="TEAM_POOL_COMPLETE", teams=args.teams,
        seed_start=args.seed_start, sha256=digest, bytes=args.out.stat().st_size,
        target85_move_count=header["target85_move_count"],
        unique_moves=len(header["move_frequency"]), transition_implemented=False,
        games_per_second=None), sort_keys=True))


if __name__ == "__main__":
    main()
