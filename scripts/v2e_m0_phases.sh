#!/bin/bash
# M0a pod driver (gpu_engine.v2e_phases): spec, CPU tests, gates, timings and the float-site report.
#
# usage: scripts/v2e_m0_phases.sh <step> <config> <reference-run> <shards-dir> <build-dir> <out-root> [limit]
#   step: spec | tests | tap_identity | chain | phases | timings | floats | all
#   limit: shards for tap_identity/chain (0 = all, the gate); phases always use min(limit, 100) or 100 when limit is 0.
# Run from the repository root with $PY set to the venv python; every output goes to a new directory under <out-root>.
# Gates run sequentially here: the first XLA compile of the full tapped oracle per lane count (~8 min) lands in the
# persistent compilation cache when one is configured, and the later gates reuse it.
set -euo pipefail
STEP=$1; CFG=$2; RUN=$3; SHARDS=$4; BUILD=$5; OUT=$6; LIMIT=${7:-0}
PY=${PY:-python}
COMMON=(--config "$CFG" --reference-run "$RUN")
DIFF=("${COMMON[@]}" --shards "$SHARDS" --build-dir "$BUILD")
PHASE_LIMIT=$(( LIMIT > 0 && LIMIT < 100 ? LIMIT : 100 ))
mkdir -p "$OUT"
run() { echo "== $*"; "$@"; }
case "$STEP" in
  spec) run "$PY" -m gpu_engine.v2e_phases spec "${COMMON[@]}" --report "$OUT/spec-report" --sub ;;
  tests) JAX_PLATFORMS=cpu run "$PY" -m unittest tests.test_gpu_engine_v2e_phases tests.test_gpu_engine_v2e_layout ;;
  tap_identity) run "$PY" -m gpu_engine.v2e_diff tap_identity "${DIFF[@]}" --limit "$LIMIT" --out "$OUT/tap-identity" ;;
  chain) run "$PY" -m gpu_engine.v2e_diff chain:gen-A-G "${DIFF[@]}" --limit "$LIMIT" --out "$OUT/chain-gen-A-G" ;;
  phases) run "$PY" -m gpu_engine.v2e_diff phase:A phase:BC phase:D phase:E0 phase:E_order0 phase:E_order1 phase:F phase:G phase:FG \
      "${DIFF[@]}" --limit "$PHASE_LIMIT" --out "$OUT/phases" ;;
  timings) run "$PY" -m gpu_engine.v2e_phases timings "${COMMON[@]}" --shards "$SHARDS" --build-dir "$BUILD" --live --out "$OUT/timings" ;;
  floats) run "$PY" -m gpu_engine.v2e_phases floats "${COMMON[@]}" --out "$OUT/floats" ;;
  all) for s in tests tap_identity phases chain timings floats; do "$0" "$s" "$CFG" "$RUN" "$SHARDS" "$BUILD" "$OUT" "$LIMIT"; done ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
