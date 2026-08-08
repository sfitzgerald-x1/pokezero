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
ACTIVE="m50-ep7 l200-ep7-wu75 v22-lr3m v3-k16 v3-k32 v3-k64 v3-k64-enthalf v3-k64-eps-entq v3-k0-enthalf v3-k1-enthalf v3-k8-enthalf v4-enthalf v4-entfull"   # v22-flat2m fork collapsed — dropped

emit_tasks() {
  # Phase-2 500k: self + foul-play per lineage (v22-flat2m forks at 2M, so it has no 500k point)
  for lin in $ACTIVE; do
    for opp in self foulplay; do
      d="$TRAITS_ROOT/traits/phase2/$lin/$opp"
      ls "$d"/events-*.jsonl.gz >/dev/null 2>&1 && shard_set_complete "$d" \
        && printf '%s %s %s %s\n' "$d" "$lin" 500000 "$opp"
    done
  done
  # Milestone tree: self and (where run) foul-play per (lineage, milestone)
  for opp in self foulplay; do
    for lin in $ACTIVE; do
      for d in $TRAITS_ROOT/traits/phase1/$lin/*/$opp; do
        [ -d "$d" ] || continue
        ls "$d"/events-*.jsonl.gz >/dev/null 2>&1 || continue
        shard_set_complete "$d" || continue
        mk=$(basename "$(dirname "$d")")            # e.g. 0100k
        printf '%s %s %s %s\n' "$d" "$lin" "$(( 10#${mk%k} * 1000 ))" "$opp"
      done
    done
  done
}

# An events-*.jsonl.gz file is CREATED when a shard starts, not when it finishes, so file presence
# says nothing about completeness. Extracting mid-write yields a set with a handful of games that
# still looks like a valid metrics file, which then silently becomes a data point in the report.
# The shard's own log records the finished game count, so require that line for every shard.
shard_set_complete() {
  local d="$1" n_ev n_done
  n_ev=$(ls "$d"/events-*.jsonl.gz 2>/dev/null | wc -l)
  n_done=$(grep -l '^WROTE .* games=' "$d"/log-*.txt 2>/dev/null | wc -l)
  [ "$n_ev" -gt 0 ] && [ "$n_done" -ge "$n_ev" ] && return 0
  # Short of a full set. That is either a sweep still in flight (must NOT be extracted) or a shard
  # that died and never will finish — foul-play shards crash at a few percent, and those sets were
  # accepted with the games that did land. Write recency separates the two: if nothing has been
  # written for a while, whatever is on disk is final and extracting it is correct.
  if [ -z "$(find "$d" -name 'events-*.jsonl.gz' -newermt '-20 minutes' 2>/dev/null)" ]; then
    [ "$n_done" -lt "$n_ev" ] && echo "  note: $d settled with $n_done/$n_ev shards finished" >&2
    return 0
  fi
  echo "  in flight (skipping): $d ($n_done/$n_ev shards finished, still being written)" >&2
  return 1
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
  # Report failures rather than dropping them: the extractor now refuses to write a metrics file
  # when its inputs are missing or parse to 0 games, and a silently skipped set would otherwise
  # look identical to one that was never due.
  if err=$(python3 "$SCR/trait_extract.py" --events "$1/events-*.jsonl.gz" --lineage "$2" \
             --milestone "$3" --out "$out" 2>&1 >/dev/null); then
    echo "ok $2 $3 $4"
  else
    echo "fail $2 $3 $4: $(echo "$err" | tail -1)"
  fi
}
export -f run_one

# Default parallelism from the CGROUP cpu quota, NEVER nproc: this runs inside a pod whose limit
# is far below the node's core count (nproc reports node cores), and oversubscribing a CPU-capped
# pod is what caused the 10x slowdown that timed out earlier sweeps.
#
# cpu.max is not always readable at the root of the cgroup mount -- on some hosts the container
# only sees it under its own cgroup path from /proc/self/cgroup, so try that first. When the quota
# cannot be read at all, fall back to TRAIT_CPUS (or a conservative constant), never to nproc:
# guessing low costs wall-clock, guessing high costs an order of magnitude to thrash.
cpu_quota() {
  local q p cg f
  cg=$(awk -F: '/^0::/{print $3}' /proc/self/cgroup 2>/dev/null)
  for f in "/sys/fs/cgroup${cg}/cpu.max" /sys/fs/cgroup/cpu.max; do   # cgroup v2: "<quota|max> <period>"
    [ -r "$f" ] || continue
    read -r q p < "$f"
    [ "$q" = "max" ] && continue
    echo $(( q / p )); return 0
  done
  for f in /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us; do  # v1
    [ -r "$f" ] || continue
    q=$(cat "$f"); p=$(cat "$(dirname "$f")/cpu.cfs_period_us")
    [ "$q" -le 0 ] && continue
    echo $(( q / p )); return 0
  done
  return 1
}
DEFAULT_JOBS=$(cpu_quota || echo "${TRAIT_CPUS:-8}")
[ "$DEFAULT_JOBS" -lt 4 ] && DEFAULT_JOBS=4
tasks=$(emit_tasks); n=$(echo "$tasks" | grep -c .)
echo "extracting $n metric sets with ${JOBS:-$DEFAULT_JOBS} workers (FORCE=${FORCE:-0}) ..."
results=$(echo "$tasks" | xargs -P "${JOBS:-$DEFAULT_JOBS}" -L1 bash -c 'run_one "$@"' _)
nfail=$(echo "$results" | grep -c '^fail ' || true)
echo "  extracted: $(echo "$results" | grep -c '^ok ')  skipped-current: $(echo "$results" | grep -c '^skip ')  failed: $nfail"
[ "$nfail" -gt 0 ] && echo "$results" | grep '^fail ' | sed 's/^/    /'

# --- FoulPlay's own play, as a contrast lineage -------------------------------------------------
# Same stored games, read from the OTHER seat (--measure-seat opponent), pooled across every active
# lineage at a milestone. This is what FoulPlay itself does at that point in the grid, so the report
# can contrast the bot's tendencies against its opponent's rather than against nothing.
#
# This pass used to be run by hand and silently went stale whenever new fp games landed. It is
# driven off the same completeness-gated task list as everything else so it cannot drift again.
# Scoped to the v3 arms on purpose: this line is read inside the v3/v4 reports as "what FoulPlay
# does against THIS generation". Pooling the v2 arms in would silently change what the contrast
# means and would invent milestones (800k, 1.9M) where only a v2 arm has foul-play games.
fp_tasks=$(echo "$tasks" | awk '$4=="foulplay" && $2 ~ /^v3-/')
fp_ms=$(echo "$fp_tasks" | awk '{print $3}' | sort -un)
fp_built=0
for ms in $fp_ms; do
  dirs=$(echo "$fp_tasks" | awk -v m="$ms" '$3==m {print $1}')
  n_lin=$(echo "$dirs" | grep -c .)
  out="$REPORT/metrics-v3-foulplay-$ms-foulplay.json"
  # Rebuild when any contributing set is newer than the pooled file: its composition changes as
  # arms retire (five arms below 3M, two above), and a stale pool would misattribute the mix.
  if [ -z "${FORCE:-}" ] && [ -f "$out" ]; then
    newest=$(echo "$dirs" | xargs -I{} ls -t {}/events-*.jsonl.gz 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
    [ -n "$newest" ] && [ ! "$newest" -nt "$out" ] && continue
  fi
  # shellcheck disable=SC2046
  if python3 "$SCR/trait_extract.py" --measure-seat opponent \
       --events $(echo "$dirs" | sed 's#$#/events-*.jsonl.gz#' | tr '\n' ' ') \
       --lineage v3-foulplay --milestone "$ms" --out "$out" >/dev/null 2>&1; then
    fp_built=$((fp_built+1))
    echo "  foulplay-seat $ms: pooled $n_lin lineage(s)"
  else
    echo "  foulplay-seat $ms: FAILED (pooling $n_lin lineage(s))"
  fi
done
echo "  foulplay-seat sets rebuilt: $fp_built"

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
