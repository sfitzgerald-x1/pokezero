#!/bin/bash
# Extract all trait metrics (Phase-1 milestones + Phase-2 500k self/foul) and build the report.
# Runs on the devbox (CPU only). Idempotent: re-run any time to refresh with more-complete data.
set -u
SCR=/shared/traits/scripts
REPORT=/shared/traits/report
mkdir -p "$REPORT"
export PYTHONPATH=/shared/traits/pokezero-src
n=0
# Phase-2 500k: self + foul-play per lineage
for lin in m50-ep7 l200-ep7-wu75 v22-lr3m m50-seq l200-seq; do
  for opp in self foulplay; do
    d="/shared/traits/phase2/$lin/$opp"
    ls "$d"/events-*.jsonl.gz >/dev/null 2>&1 || continue
    python3 "$SCR/trait_extract.py" --events "$d/events-*.jsonl.gz" --lineage "$lin" --milestone 500000 \
      --out "$REPORT/metrics-$lin-500000-$opp.json" >/dev/null 2>&1 && n=$((n+1))
  done
done
# Milestone tree: self and (where run, e.g. frontier checkpoints) foul-play per (lineage, milestone)
for opp in self foulplay; do
  for d in /shared/traits/phase1/*/*/$opp; do
    [ -d "$d" ] || continue
    lin=$(echo "$d" | cut -d/ -f5); mk=$(echo "$d" | cut -d/ -f6)   # e.g. 0100k
    ms=$(( 10#${mk%k} * 1000 ))
    ls "$d"/events-*.jsonl.gz >/dev/null 2>&1 || continue
    python3 "$SCR/trait_extract.py" --events "$d/events-*.jsonl.gz" --lineage "$lin" --milestone "$ms" \
      --out "$REPORT/metrics-$lin-$ms-$opp.json" >/dev/null 2>&1 && n=$((n+1))
  done
done
python3 "$SCR/trait_report.py" --metrics-dir "$REPORT" --out "$REPORT/trait_report.html"
echo "EXTRACTED $n metric sets -> $REPORT/trait_report.html"
