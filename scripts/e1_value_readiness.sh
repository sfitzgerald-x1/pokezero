#!/usr/bin/env bash
# E1 — value-head search-readiness (docs/mcts_design.md, "E1 first").
#
# Measures leaf-value ranking (Pearson) + calibration (ECE) for the candidate
# checkpoints on FRESH independent self-play rollouts. The roadmap's cited
# ~0.12 Pearson came from June-27 local anti-aggression/teacher-cut smokes,
# NOT from these candidates — this run produces the number that actually
# gates E0.
#
# Per candidate:
#   1. collect GAMES_PER_SPLIT self-play games (fit split, seed band A)
#   2. collect GAMES_PER_SPLIT self-play games (held-out split, seed band B)
#   3. value-calibration-compare: raw vs affine vs isotonic on held-out
# Then (best-effort) a cross-pool read: every head scored on the belief-1.5M
# held-out pool for apples-to-apples ranking. Encoder mismatches are
# tolerated and logged, not fatal.
#
# Usage (from repo root, venv synced):
#   uv sync                       # once
#   bash scripts/e1_value_readiness.sh
# Env overrides:
#   GAMES_PER_SPLIT (default 32)  SEED_BASE (default 20260703000)
#   OUT_ROOT (default runs/e1-value-readiness-20260703)
#   POKEZERO_SHOWDOWN_ROOT (defaults per local_showdown.py)
#
# Search-readiness bar (mcts_design.md): held-out Pearson >= 0.3 for at
# least one transform. The script reports; it does not hard-fail, because a
# failed E1 is itself the finding (value work precedes any strength claim).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GAMES_PER_SPLIT="${GAMES_PER_SPLIT:-32}"
SEED_BASE="${SEED_BASE:-20260703000}"
OUT_ROOT="${OUT_ROOT:-runs/e1-value-readiness-20260703}"
PEARSON_BAR="0.30"

# candidate label -> checkpoint path
CANDIDATES=(
  "belief-1-5m:checkpoints/pokezero-belief-gen3-1-5m.pt"
  "fpdistill-1-5m:checkpoints/fpdistill-gen3-1-5m.pt"
  "fpdistill-1m:checkpoints/pokezero-fpdistill-gen3-1m.pt"
  "no-belief-1-5m:checkpoints/pokezero-no-belief-gen3-1-5m.pt"
)

mkdir -p "$OUT_ROOT"
echo "E1 value-readiness run -> $OUT_ROOT (games/split=$GAMES_PER_SPLIT, seed base=$SEED_BASE)"

offset=0
for entry in "${CANDIDATES[@]}"; do
  label="${entry%%:*}"
  ckpt="${entry#*:}"
  if [[ ! -f "$ckpt" ]]; then
    echo "[skip] $label: missing checkpoint $ckpt" | tee -a "$OUT_ROOT/warnings.log"
    offset=$((offset + 1))
    continue
  fi
  dir="$OUT_ROOT/$label"
  mkdir -p "$dir"
  fit_seed=$((SEED_BASE + offset * 10000))
  heldout_seed=$((SEED_BASE + offset * 10000 + 5000))

  echo "[$label] collecting fit split ($GAMES_PER_SPLIT games, seeds $fit_seed+)"
  uv run pokezero-rollout collect \
    --games "$GAMES_PER_SPLIT" \
    --out "$dir/fit-rollouts.jsonl" \
    --p1-policy "neural:$ckpt" \
    --p2-policy "neural:$ckpt" \
    --seed-start "$fit_seed"

  echo "[$label] collecting held-out split ($GAMES_PER_SPLIT games, seeds $heldout_seed+)"
  uv run pokezero-rollout collect \
    --games "$GAMES_PER_SPLIT" \
    --out "$dir/heldout-rollouts.jsonl" \
    --p1-policy "neural:$ckpt" \
    --p2-policy "neural:$ckpt" \
    --seed-start "$heldout_seed"

  echo "[$label] value-calibration-compare (raw vs affine vs isotonic on held-out)"
  uv run pokezero-neural value-calibration-compare \
    --checkpoint "$ckpt" \
    --data "$dir/fit-rollouts.jsonl" \
    --eval-data "$dir/heldout-rollouts.jsonl" \
    --selection-metric pearson_correlation \
    --out "$dir/calibration-compare.json"

  offset=$((offset + 1))
done

# Cross-pool read: every head on the belief-1.5M held-out pool (best-effort).
POOL="$OUT_ROOT/belief-1-5m/heldout-rollouts.jsonl"
if [[ -f "$POOL" ]]; then
  for entry in "${CANDIDATES[@]}"; do
    label="${entry%%:*}"
    ckpt="${entry#*:}"
    [[ -f "$ckpt" && "$label" != "belief-1-5m" ]] || continue
    echo "[cross-pool] $label on belief-1.5M held-out pool"
    if ! uv run pokezero-neural value-calibration \
      --checkpoint "$ckpt" \
      --data "$POOL" \
      --json > "$OUT_ROOT/$label/cross-pool-belief15m.json" 2> "$OUT_ROOT/$label/cross-pool-belief15m.err"; then
      echo "[cross-pool] $label FAILED (likely encoder mismatch) — see $OUT_ROOT/$label/cross-pool-belief15m.err" \
        | tee -a "$OUT_ROOT/warnings.log"
    fi
  done
fi

echo
echo "=== E1 summary (bar: held-out Pearson >= $PEARSON_BAR on any transform) ==="
python3 - "$OUT_ROOT" "$PEARSON_BAR" <<'PY'
import json, sys
from pathlib import Path

root, bar = Path(sys.argv[1]), float(sys.argv[2])

def metric(entry, name):
    for scope in (entry, entry.get("report") or {}):
        if isinstance(scope, dict) and scope.get(name) is not None:
            return scope[name]
    return None

for compare in sorted(root.glob("*/calibration-compare.json")):
    label = compare.parent.name
    try:
        payload = json.loads(compare.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"{label:16s} unreadable: {exc}")
        continue
    best_pearson = None
    for entry in payload.get("methods", []):
        pearson = metric(entry, "pearson_correlation")
        ece = metric(entry, "expected_calibration_error")
        sign = metric(entry, "sign_accuracy")
        fmt = lambda v: "n/a" if v is None else f"{float(v):.4f}"
        print(
            f"{label:16s} {entry.get('method', '?'):9s} "
            f"pearson={fmt(pearson)} ece={fmt(ece)} sign={fmt(sign)}"
        )
        if pearson is not None:
            best_pearson = max(best_pearson or -2.0, float(pearson))
    verdict = (
        "UNKNOWN" if best_pearson is None
        else ("SEARCH-READY (clears bar)" if best_pearson >= bar else "NOT search-ready")
    )
    print(f"{label:16s} best_method={payload.get('best_method')} -> {verdict}")
    print("-" * 72)
PY
echo "Done. Artifacts under $OUT_ROOT/."
