#!/bin/bash
# Extract all trait metrics (Phase-1 milestones + Phase-2 500k self/foul) and build the report.
# Runs on the devbox (CPU only). Idempotent: re-run any time to refresh with more-complete data.
# Extractions run in parallel (JOBS, default 10); only active lineages are extracted (the seq
# lineages are dropped from the report entirely). The report build runs after all extractions.
set -u
SCR=/shared/traits/scripts
REPORT=/shared/traits/report
mkdir -p "$REPORT"
export SCR REPORT
export PYTHONPATH=/shared/traits/pokezero-src
ACTIVE="m50-ep7 l200-ep7-wu75 v22-lr3m v3-k16 v3-k32 v3-k64"   # v22-flat2m fork collapsed — dropped

emit_tasks() {
  # Phase-2 500k: self + foul-play per lineage (v22-flat2m forks at 2M, so it has no 500k point)
  for lin in $ACTIVE; do
    for opp in self foulplay; do
      d="/shared/traits/phase2/$lin/$opp"
      ls "$d"/events-*.jsonl.gz >/dev/null 2>&1 && printf '%s %s %s %s\n' "$d" "$lin" 500000 "$opp"
    done
  done
  # Milestone tree: self and (where run) foul-play per (lineage, milestone)
  for opp in self foulplay; do
    for lin in $ACTIVE; do
      for d in /shared/traits/phase1/$lin/*/$opp; do
        [ -d "$d" ] || continue
        ls "$d"/events-*.jsonl.gz >/dev/null 2>&1 || continue
        mk=$(basename "$(dirname "$d")")            # e.g. 0100k
        printf '%s %s %s %s\n' "$d" "$lin" "$(( 10#${mk%k} * 1000 ))" "$opp"
      done
    done
  done
}

run_one() {   # args: dir lineage milestone opp
  python3 "$SCR/trait_extract.py" --events "$1/events-*.jsonl.gz" --lineage "$2" --milestone "$3" \
    --out "$REPORT/metrics-$2-$3-$4.json" >/dev/null 2>&1 && echo "ok $2 $3 $4"
}
export -f run_one

tasks=$(emit_tasks); n=$(echo "$tasks" | grep -c .)
echo "extracting $n metric sets with ${JOBS:-10} workers ..."
echo "$tasks" | xargs -P "${JOBS:-10}" -L1 bash -c 'run_one "$@"' _ | grep -c '^ok ' | xargs echo "completed:"
# v2 report (m50-ep7 / l200-ep7-wu75 / v22-lr3m) and the separate v3 report (empty until v3 runs
# exist and V3_LINEAGES/ACTIVE are populated).
python3 "$SCR/trait_report.py" --metrics-dir "$REPORT" --out "$REPORT/trait_report.html" --set v2
python3 "$SCR/trait_report.py" --metrics-dir "$REPORT" --out "$REPORT/trait_report_v3.html" --set v3
echo "EXTRACTED $n metric sets -> $REPORT/trait_report.html (+ trait_report_v3.html)"
