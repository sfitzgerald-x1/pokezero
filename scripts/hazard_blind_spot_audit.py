"""Run the public-only Step 3 Spikes/Rapid Spin blind-spot audit.

The output contains compact public state descriptors bound to the canonical
replay corpus, paired deterministic and explicit audit-only Dirichlet PUCT
sweeps at legal+{0,24,120}, and the predefined E, R_off, and DeltaChoice_on
metrics. It never reads the source battle's opponent request or opponent legal
action mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pokezero.hazard_audit import (
    AuditConfig,
    PublicBeliefWorldProvider,
    canonical_hash,
    capture_hazard_audit_corpus,
    iter_hazard_audit_decisions_from_public_corpus,
    run_hazard_blind_spot_audit,
    sha256_file,
)
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.neural_policy import (
    evaluate_transformer_action_priors,
    evaluate_transformer_observation_value,
)
from pokezero.online_client import build_agent
from pokezero.opponents import require_current_family_checkpoint_paths
from pokezero.policy import MaxDamagePolicy
from pokezero.public_decision_corpus import (
    PublicDecisionCorpusWriter,
    open_public_decision_corpus,
    public_corpus_manifest,
)
from pokezero.randbat import load_gen3_randbat_source_cached


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--showdown-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--public-corpus",
        type=Path,
        help="Step 2 pokezero.public-decision-corpus.v1 JSONL. When set, fixed-driver capture is skipped.",
    )
    parser.add_argument(
        "--fixed-driver-corpus-out",
        type=Path,
        help="Canonical public JSONL sidecar for fixed-driver capture; defaults beside --out.",
    )
    parser.add_argument(
        "--max-public-decisions",
        type=int,
        default=None,
        help="Optional deterministic prefix cap when reading --public-corpus.",
    )
    parser.add_argument("--games", type=int, default=60, help="fixed-driver games used to build the corpus")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=800)
    parser.add_argument("--max-decision-rounds", type=int, default=250)
    parser.add_argument("--cpuct", type=float, default=1.25)
    parser.add_argument("--low-prior-threshold", type=float, default=0.01)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-mix", type=float, default=0.25)
    parser.add_argument("--dirichlet-seed", type=int, default=20260710)
    parser.add_argument("--belief-world-sample-cap", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    require_current_family_checkpoint_paths((args.checkpoint,), context="hazard blind-spot audit")
    agent = build_agent(args.checkpoint, args.showdown_root, our_name="hazard-audit", deterministic=True)
    feature_masks = getattr(agent, "feature_masks", None)
    env_config_kwargs = {
        "showdown_root": args.showdown_root,
        "observation_spec": agent.spec,
        "category_vocab": agent.vocab,
        "set_belief_source": True,
    }
    if feature_masks is not None:
        env_config_kwargs["feature_masks"] = feature_masks
    env_config = LocalShowdownConfig(**env_config_kwargs)
    if args.public_corpus is not None:
        if args.fixed_driver_corpus_out is not None:
            parser.error("--fixed-driver-corpus-out is only valid without --public-corpus")
        public_corpus = open_public_decision_corpus(
            args.public_corpus,
            max_decisions=args.max_public_decisions,
        )
        corpus = iter_hazard_audit_decisions_from_public_corpus(public_corpus)
    else:
        corpus = capture_hazard_audit_corpus(
            env_config=env_config,
            games=args.games,
            seed_start=args.seed_start,
            max_states=args.max_states,
            max_decision_rounds=args.max_decision_rounds,
        )
        fixed_driver_corpus_config = {
            "source": "fixed-checkpoint-independent-drivers",
            "drivers": [
                "max-damage-vs-max-damage",
                "max-damage-vs-random-legal",
                "simple-legal-vs-simple-legal",
                "random-legal-vs-random-legal",
            ],
            "games": args.games,
            "seed_start": args.seed_start,
            "max_states": args.max_states,
            "max_decision_rounds": args.max_decision_rounds,
        }
    config = AuditConfig(
        cpuct=args.cpuct,
        low_prior_threshold=args.low_prior_threshold,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_mix=args.dirichlet_mix,
        dirichlet_seed=args.dirichlet_seed,
    )

    def priors(history):
        return tuple(
            evaluate_transformer_action_priors(
                model=agent.policy.model,
                result=agent.policy.result,
                observations=history,
                device=args.device,
            )
        )

    def value(history):
        return evaluate_transformer_observation_value(
            model=agent.policy.model,
            result=agent.policy.result,
            observations=history,
            device=args.device,
        )

    set_source = load_gen3_randbat_source_cached(args.showdown_root)
    fixed_driver_corpus_path: Path | None = None
    if args.public_corpus is None:
        fixed_driver_corpus_path = args.fixed_driver_corpus_out or args.out.with_name(
            f"{args.out.stem}.public-decisions.jsonl"
        )
        if fixed_driver_corpus_path == args.out:
            parser.error("--fixed-driver-corpus-out must not equal --out")
        fixed_driver_corpus_config = {
            **fixed_driver_corpus_config,
            "opponent_legal_mask_mode": "hidden",
            "root_dirichlet_alpha": None,
            "selected_hazard_state_ids": [decision.state_id for decision in corpus],
        }
        manifest = public_corpus_manifest(
            checkpoint_sha256="fixed-checkpoint-independent",
            belief_set_source_hash=set_source.metadata.source_hash,
            capture_config=fixed_driver_corpus_config,
        )
        with PublicDecisionCorpusWriter(fixed_driver_corpus_path, manifest=manifest) as writer:
            for decision in corpus:
                writer.append(decision.public_record)

    def audit_provenance() -> dict[str, object]:
        if args.public_corpus is not None:
            corpus_config: dict[str, object] = {
                "source": "pokezero.public-decision-corpus.v1",
                "path": str(args.public_corpus),
                "source_file_sha256": public_corpus.source_file_sha256,
                "selected_content_sha256": public_corpus.selected_content_sha256,
                "selection": {
                    "max_decisions": args.max_public_decisions,
                    "selected_decision_count": public_corpus.selected_decision_count,
                },
                "schema_version": public_corpus.manifest.get("schema_version"),
            }
        else:
            assert fixed_driver_corpus_path is not None
            corpus_config = {
                "source": "pokezero.public-decision-corpus.v1",
                "path": str(fixed_driver_corpus_path),
                "source_file_sha256": sha256_file(fixed_driver_corpus_path),
                "selection": {
                    "selected_hazard_state_count": len(corpus),
                },
                "capture": fixed_driver_corpus_config,
            }
        return {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_model_config_hash": canonical_hash(agent.policy.result.model_config),
            "showdown_root": str(args.showdown_root.resolve()),
            "randbat_source_hash": set_source.metadata.source_hash,
            "sampled_world_opponent_policy": "max-damage",
            "corpus_config": corpus_config,
        }

    payload = run_hazard_blind_spot_audit(
        decisions=corpus,
        env_factory=lambda: LocalShowdownEnv(env_config),
        action_priors=priors,
        value_fn=value,
        world_provider=PublicBeliefWorldProvider(
            env_factory=lambda: LocalShowdownEnv(env_config),
            set_source=set_source,
            sampled_world_opponent_policy=MaxDamagePolicy(showdown_root=args.showdown_root),
            world_sample_cap=args.belief_world_sample_cap,
        ),
        config=config,
        provenance_factory=audit_provenance,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"hazard_blind_spot_audit: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
