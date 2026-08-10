#!/usr/bin/env bash
# The instrument must be able to report failure -- run this before trusting a zero.
#
# COMMITTED deliberately. This script is the evidence for every zero the census
# publishes, and independent review could not re-run it because the original was
# headed "NOT COMMITTED": no reviewer and no CI could reproduce the forced-failure
# table. PLAN section 3 assigns the mutation battery to this harness specifically.
#
# Four arms over the same two plan games at the same budget. Each of construct /
# abort / unmapped must produce 100% attributed truth rejections; `none` must
# produce zero; and the DRIVER counters must be identical in all four, which is
# what proves the forcing did not leak into the thing being measured.
# Same 2 games, same seeds, same budget, four arms.
set -u
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
ART="${ART:?set ART to the directory holding model_ts.pt, the checkpoint and encoder_tables.json}"
OUT="${OUT:-$REPO/_local/instrument}"
mkdir -p "$OUT"
export PYTHONPATH=$REPO/src PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for MODE in none construct abort unmapped; do
  "$REPO/.venv/bin/python" -B "$REPO/scripts/truth_differential_census.py" \
    --mode run --plan "$REPO/_local/census-plan.json" --max-games 2 \
    --model-path "$ART/.mcts-eval-artifacts-054071e376d133a0/model_ts.pt" \
    --checkpoint "$ART/v3hist-k64-enthalf-5m-20260723-iteration-2657.pt" \
    --tables "$ART/.mcts-eval-artifacts-054071e376d133a0/encoder_tables.json" \
    --driver-leaf-eval hp_fraction_crate --driver-worlds 8 --driver-sims 256 \
    --truth-sims 8 --truth-depth 4 --truth-batch 8 --max-rounds 250 \
    --force "$MODE" --tag "force-$MODE" --out "$OUT/$MODE.json" \
    > "$OUT/$MODE.log" 2>&1 &
done
wait
echo INSTRUMENT_TEST_DONE
