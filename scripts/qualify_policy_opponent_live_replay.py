#!/usr/bin/env python3
"""Validate frozen public replay only; stdout is not an experiment/strength PASS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcts_mcts_h2h import _showdown_source_provenance
from pokezero.collection import env_config_with_policy_spec_masks
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.mcts_eval.policy_opponent_live_replay import qualify_live_replay
from pokezero.mcts_eval.resolver import ContractError, resolve_checkpoint_contract, sha256_file
from pokezero.neural_policy import load_transformer_checkpoint_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("roster", "roster-sha256", "source-root", "checkpoint",
                 "checkpoint-sha256", "showdown-root", "showdown-runtime-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    try:
        runtime = _showdown_source_provenance(args.showdown_root)
        if runtime["content_sha256"] != args.showdown_runtime_sha256:
            raise ContractError("live replay Showdown runtime bytes differ from registered hash")
        contract = resolve_checkpoint_contract(args.checkpoint, expected_sha256=args.checkpoint_sha256,
            showdown_root=args.showdown_root, showdown_source_sha256=runtime["content_sha256"])
        payload = load_transformer_checkpoint_payload(args.checkpoint)
        belief_hash = payload.get("belief_set_source_hash")
        if not isinstance(belief_hash, str) or not belief_hash:
            raise ContractError("checkpoint lacks its registered belief-source hash")
        config = env_config_with_policy_spec_masks(
            LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True),
            [f"neural:{args.checkpoint}"], context="paper frozen-root live replay preflight")
        with LocalShowdownEnv(config) as env:
            if env.belief_set_source_hash != belief_hash:
                raise ContractError("live replay belief source differs from checkpoint")
            result = qualify_live_replay(args.roster, expected_roster_sha256=args.roster_sha256,
                                        source_root=args.source_root, env=env)
        if (_showdown_source_provenance(args.showdown_root) != runtime
                or sha256_file(args.checkpoint) != args.checkpoint_sha256):
            raise ContractError("live replay runtime/checkpoint drift during qualification")
        result.update(checkpoint=contract.to_manifest(), showdown_runtime=runtime,
                      belief_source_hash=belief_hash, immutable_image_qualified=False)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"LIVE REPLAY PREFLIGHT REFUSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
