#!/bin/bash
# Extract all trait metrics (Phase-1 milestones + Phase-2 500k self/foul) and build the report.
# Runs on the devbox (CPU only). Idempotent: re-run any time to refresh with more-complete data.
# Extractions run in parallel (JOBS, default 10); only active lineages are extracted (the seq
# lineages are dropped from the report entirely). The report build runs after all extractions.
set -u
# Storage root is configurable so the pipeline runs from any devbox. The box that owns the shared
# volume mounts it at <private-store>; a box that CROSS-MOUNTS the same volume sees it at a
# different path. Point TRAITS_ROOT at whichever applies; everything below is relative to it.
TRAITS_ROOT="${TRAITS_ROOT:-<private-store>}"
[ -d "$TRAITS_ROOT/traits" ] || { echo "TRAITS_ROOT=$TRAITS_ROOT has no traits/ dir" >&2; exit 2; }
SCR=$TRAITS_ROOT/traits/scripts
REPORT=$TRAITS_ROOT/traits/report
mkdir -p "$REPORT"
export SCR REPORT
export PYTHONPATH=$TRAITS_ROOT/traits/pokezero-src
ACTIVE="m50-ep7 l200-ep7-wu75 v22-lr3m v3-k16 v3-k32 v3-k64 v3-k64-enthalf v3-k64-eps-entq v3-k0-enthalf v3-k1-enthalf v3-k8-enthalf"   # v22-flat2m fork collapsed — dropped

emit_tasks() {
  # Phase-2 500k: self + foul-play per lineage (v22-flat2m forks at 2M, so it has no 500k point)
  for lin in $ACTIVE; do
    for opp in self foulplay; do
      d="$TRAITS_ROOT/traits/phase2/$lin/$opp"
      ls "$d"/events-*.jsonl.gz >/dev/null 2>&1 && printf '%s %s %s %s\n' "$d" "$lin" 500000 "$opp"
    done
  done
  # Milestone tree: self and (where run) foul-play per (lineage, milestone)
  for opp in self foulplay; do
    for lin in $ACTIVE; do
      for d in $TRAITS_ROOT/traits/phase1/$lin/*/$opp; do
        [ -d "$d" ] || continue
        ls "$d"/events-*.jsonl.gz >/dev/null 2>&1 || continue
        mk=$(basename "$(dirname "$d")")            # e.g. 0100k
        printf '%s %s %s %s\n' "$d" "$lin" "$(( 10#${mk%k} * 1000 ))" "$opp"
      done
    done
  done
}

# Re-extraction is the expensive step: ~6s of single-core protocol-walk per (lineage, milestone)
# — I/O is negligible, it is GameParse. A routine refresh only has a handful of NEW milestones,
# so by default skip any metrics file that is already newer than every event file feeding it.
# Set FORCE=1 after a metric-DEFINITION change (new category/field), which needs a real backfill.
run_one() {   # args: dir lineage milestone opp
  out="$REPORT/metrics-$2-$3-$4.json"
  if [ -z "${FORCE:-}" ] && [ -f "$out" ]; then
    newest=$(ls -t "$1"/events-*.jsonl.gz 2>/dev/null | head -1)
    if [ -n "$newest" ] && [ ! "$newest" -nt "$out" ]; then echo "skip $2 $3 $4"; return; fi
  fi
  python3 "$SCR/trait_extract.py" --events "$1/events-*.jsonl.gz" --lineage "$2" --milestone "$3" \
    --out "$out" >/dev/null 2>&1 && echo "ok $2 $3 $4"
}
export -f run_one

# Default parallelism from the CGROUP cpu quota, never nproc: this runs inside a pod whose limit
# (16 CPU) is far below the node's core count (nproc reports 128), and oversubscribing a
# CPU-capped pod is what caused the 10x slowdown that timed out earlier sweeps.
cpu_quota() {
  local q p
  if [ -r /sys/fs/cgroup/cpu.max ]; then            # cgroup v2: "<quota|max> <period>"
    read -r q p < /sys/fs/cgroup/cpu.max
    [ "$q" = "max" ] && return 1
    echo $(( q / p )); return 0
  fi
  if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then   # cgroup v1
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [ "$q" -le 0 ] && return 1
    echo $(( q / p )); return 0
  fi
  return 1
}
DEFAULT_JOBS=$(cpu_quota || nproc 2>/dev/null || echo 8)
[ "$DEFAULT_JOBS" -lt 4 ] && DEFAULT_JOBS=4
tasks=$(emit_tasks); n=$(echo "$tasks" | grep -c .)
echo "extracting $n metric sets with ${JOBS:-$DEFAULT_JOBS} workers (FORCE=${FORCE:-0}) ..."
results=$(echo "$tasks" | xargs -P "${JOBS:-$DEFAULT_JOBS}" -L1 bash -c 'run_one "$@"' _)
echo "  extracted: $(echo "$results" | grep -c '^ok ')  skipped-current: $(echo "$results" | grep -c '^skip ')"
# v2 report (m50-ep7 / l200-ep7-wu75 / v22-lr3m) and the separate v3 report (empty until v3 runs
# exist and V3_LINEAGES/ACTIVE are populated).
python3 "$SCR/trait_report.py" --metrics-dir "$REPORT" --out "$REPORT/trait_report.html" --set v2
# Two standalone v3 reports: legacy (retired k16/k32/k64 history-length arms) and ent_fix (the
# active entropy-fix variants — enthalf, eps-entq, and any future v3 variant).
python3 "$SCR/trait_report.py" --metrics-dir "$REPORT" --out "$REPORT/trait_report_v3_legacy.html" --set v3_legacy
python3 "$SCR/trait_report.py" --metrics-dir "$REPORT" --out "$REPORT/trait_report_v3_ent_fix.html" --set v3_ent_fix
# v4: the successor generation plus the carried-over control arms (k0/k1).
python3 "$SCR/trait_report.py" --metrics-dir "$REPORT" --out "$REPORT/trait_report_v4.html" --set v4
echo "EXTRACTED $n metric sets -> trait_report.html + trait_report_v3_legacy.html + trait_report_v3_ent_fix.html"
