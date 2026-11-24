"""Assemble a digest-bound constructor roster from ordered shared-storage shards."""

import argparse
import json
from pathlib import Path

from gpu_engine.v2_pool_assembly import assemble, file_sha, generate_gender_policy, subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gender-policy-out', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_bytes())
    result = assemble(manifest, args.out)
    if args.gender_policy_out is not None:
        result['gender_policy'] = generate_gender_policy(result, manifest['showdown'], args.gender_policy_out)
    peak = Path('/sys/fs/cgroup/memory.peak')
    if peak.exists():
        result['memory_peak_bytes'] = int(peak.read_text())
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
