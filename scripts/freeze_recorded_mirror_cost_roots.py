#!/usr/bin/env python3
"""Freeze TRAIN-only saved mirror root addresses; no model/simulator/search."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokezero.mirror_value_probe import file_sha
from pokezero.recorded_mirror_roots import freeze_cost_roots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--collection-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh root-address output required; no resume or replacement")
    result = freeze_cost_roots(args.collection, expected_sha256=args.collection_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="FROZEN_COST_ADDRESSES_NOT_ADMITTED", count=result["count"],
        output_sha256=file_sha(args.output), selected_root_sha256=result["selected_root_sha256"],
        new_games=0, native_search_calls=0)), flush=True)
