#!/usr/bin/env python3
"""Score admitted saved actual siblings with both A1 heads; no collection/replay."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokezero.value_sibling_analysis import analyze


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "checkpoint", "summary", "head"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output path required")
    result = analyze(args.capture, args.checkpoint, args.summary, args.head,
        capture_sha256=args.capture_sha256, checkpoint_sha256=args.checkpoint_sha256,
        summary_sha256=args.summary_sha256, head_sha256=args.head_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "captured_root_contexts", "heads", "new_games", "model_fits")}))
