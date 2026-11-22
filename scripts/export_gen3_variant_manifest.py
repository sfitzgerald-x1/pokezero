#!/usr/bin/env python3
"""Export the source-bound variant order consumed by the offline Node compiler."""
import argparse
import json
from pathlib import Path

from pokezero.generator_weights import source_manifest
from pokezero.randbat import load_gen3_randbat_source_cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = load_gen3_randbat_source_cached(args.root)
    args.output.write_text(json.dumps(source_manifest(source), ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
