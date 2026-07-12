"""Stage A extractor: dump one checkpoint's policy+value fingerprint over a fixed corpus.

Streams the public-decision JSONL one record at a time (the bulk loader holds every
record's full history in memory and OOMs on the ~6.6k-record corpus). For window-1
checkpoints the forward is invariant to history (verified: current-only == full-history
to 0 ulp), so we score only the current observation — bounded memory and faster.

Reuses the vetted forward primitives; the observation-census guard fires inside the
forward if a checkpoint's schema does not match the corpus (the plan's census gate,
exercised not bypassed). Emits one JSON with, per decision: top-1 legal action, the
legal-renormalized policy distribution, and the raw value.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pokezero.actions import ACTION_COUNT
from pokezero.neural_policy import (
    load_transformer_checkpoint,
    evaluate_transformer_action_priors,
    evaluate_transformer_observation_value,
    observation_spec_from_model_config,
)
from pokezero.public_decision_corpus import PublicObservation, sha256_file
from pokezero.prior_belief_profile import _normalized_raw_legal_priors


def stream_payloads(path: Path):
    """Yield decision payloads without reconstructing the (unused, expensive) history."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if payload.get("record_type") == "manifest":
                continue
            yield payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    model, result = load_transformer_checkpoint(args.checkpoint, map_location=args.device)
    spec = observation_spec_from_model_config(result.model_config)
    print(f"[t={time.time()-t0:.0f}s] model loaded schema={spec.schema_version}", flush=True)

    rows = []
    census = None
    for i, p in enumerate(stream_payloads(args.corpus)):
        if args.limit and i >= args.limit:
            break
        legal = tuple(idx for idx, ok in enumerate(p["current_legal_action_mask"]) if ok)
        if not legal:
            continue
        # window-1: forward the current observation only (history-invariant, verified);
        # history is never parsed.
        cur = PublicObservation.from_dict(p["observation"]).to_observation(belief_view=p["public_belief_view"])
        obs = (cur,)
        if census is None:
            census = len(cur.numeric_features[0])
        raw = evaluate_transformer_action_priors(
            model=model, result=result, observations=obs, temperature=1.0, device=args.device
        )
        dist = _normalized_raw_legal_priors(raw, legal)
        value = evaluate_transformer_observation_value(
            model=model, result=result, observations=obs, device=args.device
        )
        top1 = max(legal, key=lambda a: dist[a])
        rows.append(
            {
                "decision_id": p["decision_id"],
                "battle_id": p["battle_id"],
                "seed": p["seed"],
                "acting_player": p["acting_player"],
                "legal": list(legal),
                "top1": int(top1),
                "probs": {int(a): float(dist[a]) for a in legal},
                "value": float(value),
            }
        )
        if i == 0:
            print(f"[t={time.time()-t0:.0f}s] first forward done", flush=True)
        if (i + 1) % 1000 == 0:
            print(f"  {args.label}/{args.role}: {i+1} ({time.time()-t0:.0f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "pokezero.diversity_fingerprint.v1",
        "label": args.label,
        "role": args.role,
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "observation_schema": spec.schema_version,
        "numeric_census": census,
        "corpus": str(args.corpus),
        "corpus_sha256": sha256_file(args.corpus),
        "action_count": ACTION_COUNT,
        "n_decisions": len(rows),
        "elapsed_seconds": time.time() - t0,
        "decisions": rows,
    }
    args.out.write_text(json.dumps(payload))
    print(f"WROTE {args.out} decisions={len(rows)} schema={spec.schema_version} census={census} ({payload['elapsed_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
