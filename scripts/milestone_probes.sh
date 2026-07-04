#!/usr/bin/env bash
# Milestone probe + ecology watchdog sweep over the active foundation runs
# (WS-3 items 2-3, docs/next_train_readiness_plan.md). Run manually or from a
# cron on an operator machine with kubectl access to the training cluster.
#
# Per invocation:
#   0. take an exclusive lock (runs/milestone-probes/.sweep.lock) — if another
#      sweep holds it, exit 0 immediately: overlapping sweeps must never race
#      on the ledger or tear in-flight checkpoint copies
#   1. discover running *fnd-controller-* pods -> run ids (from pod --run-id arg)
#   2. per run, per ~100k-game milestone not yet in the local ledger:
#        - kubectl cp the milestone-nearest iteration checkpoint into
#          checkpoints/curated/<run>-i<iter>.pt (byte-size verified); milestones
#          whose nearest checkpoint is >30k games away are recorded as SKIPPED
#          ledger lines instead of probing the wrong checkpoint
#        - cross-pool Pearson: pokezero-neural value-calibration on pool-self-v1
#          (and pool-fp-v1 when present)
#        - dV hazard probe: scripts/hazard_probe.py (main carries the value-
#          response section since #501; verified at startup)
#        - append one JSONL line to runs/milestone-probes/<run>/ledger.jsonl
#   3. ecology watchdogs over the run's eval-timeline.jsonl: game-length drift
#      (+50% vs the run's own 30k-100k band, falling back to the earliest rows
#      in the window for continuation runs, degrading LOUDLY when no baseline
#      exists; the first-computed baseline is persisted to
#      runs/milestone-probes/<run>/drift-baseline.json and reused by later
#      sweeps, so it cannot slide forward with the tail window), per-fidelity
#      max-damage regression (>10 points below that fidelity's run peak),
#      policy_entropy < 0.35, and a non-finite-data warning. Alarms
#      append to runs/milestone-probes/ALERTS.jsonl and print loudly.
#
# Every kubectl call carries --request-timeout and kubectl cp is additionally
# wrapped in a hard timeout, so a wedged API server or node drain cannot hang
# the sweep (and therefore cannot pile up overlapping crons behind the lock).
#
# Idempotent and resumable: probed milestones are skipped via the ledger;
# existing checkpoint copies and probe outputs are reused. Per-run failures are
# isolated — one bad run does not kill the sweep (exit 1 at the end instead).
#
# Required env (no defaults — cluster specifics never live in this repo):
#   POKEZERO_CLUSTER_CONTEXT     kubectl context of the training cluster
#   POKEZERO_CLUSTER_NAMESPACE   namespace running the foundation controller pods
#   POKEZERO_SHARED_ROOT         in-pod shared filesystem root holding <run-id>/ dirs
#   POKEZERO_SHOWDOWN_ROOT       local pokemon-showdown (sim) checkout, for the
#                                hazard-probe state corpus (not needed in --dry-run)
# Optional env:
#   POKEZERO_HAZARD_GAMES        hazard-probe corpus games            [default 150]
#   POKEZERO_MILESTONE_STEP      milestone spacing in games           [default 100000]
#   POKEZERO_MILESTONE_MAX_DIST  max |checkpoint games - milestone|   [default 30000]
#   POKEZERO_TIMELINE_TAIL       eval-timeline rows fed to watchdogs  [default 500]
#   POKEZERO_KUBECTL_TIMEOUT     --request-timeout for kubectl calls  [default 60s]
#   POKEZERO_CP_TIMEOUT          hard timeout for kubectl cp, seconds [default 1800]
#
# Usage (from repo root, `uv sync` done once):
#   scripts/milestone_probes.sh [--dry-run]
# --dry-run: read-only against the cluster AND the local tree — report runs
# discovered, pending milestones, and watchdog alarms; copy nothing, probe
# nothing, write nothing (the sweep lock is skipped: dry-run cannot corrupt).

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
  esac
done

CTX="${POKEZERO_CLUSTER_CONTEXT:?set POKEZERO_CLUSTER_CONTEXT (kubectl context)}"
NS="${POKEZERO_CLUSTER_NAMESPACE:?set POKEZERO_CLUSTER_NAMESPACE (cluster namespace)}"
SHARED="${POKEZERO_SHARED_ROOT:?set POKEZERO_SHARED_ROOT (in-pod shared filesystem root)}"
if [ "$DRY_RUN" -eq 0 ]; then
  SHOWDOWN="${POKEZERO_SHOWDOWN_ROOT:?set POKEZERO_SHOWDOWN_ROOT (pokemon-showdown checkout, for the hazard probe)}"
fi
HAZARD_GAMES="${POKEZERO_HAZARD_GAMES:-150}"
STEP="${POKEZERO_MILESTONE_STEP:-100000}"
MAX_DIST="${POKEZERO_MILESTONE_MAX_DIST:-30000}"
TIMELINE_TAIL="${POKEZERO_TIMELINE_TAIL:-500}"
REQ_TIMEOUT="${POKEZERO_KUBECTL_TIMEOUT:-60s}"
CP_TIMEOUT="${POKEZERO_CP_TIMEOUT:-1800}"

KC=(kubectl --context "$CTX" -n "$NS" --request-timeout="$REQ_TIMEOUT")
# kubectl cp streams multi-GB checkpoints: it gets its own (long) request
# timeout plus a hard wall-clock timeout via run_with_timeout below.
KC_CP=(kubectl --context "$CTX" -n "$NS" --request-timeout="${CP_TIMEOUT}s")
HELPER=(python3 "$REPO/scripts/milestone_probes.py")
HAZARD_PROBE="$REPO/scripts/hazard_probe.py"
PROBE_ROOT="$REPO/runs/milestone-probes"
ALERTS="$PROBE_ROOT/ALERTS.jsonl"
LOCKFILE="$PROBE_ROOT/.sweep.lock"
CURATED="$REPO/checkpoints/curated"
POOL_SELF="$REPO/runs/e1-value-readiness-20260703/belief-1-5m/heldout-rollouts.jsonl"
POOL_FP="$REPO/runs/pool-fp-v1-20260704/pool-fp-v1.jsonl"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/milestone-probes.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

log()   { echo "[milestone-probes] $*"; }
loud()  { echo; echo "################################################################"; printf '%s\n' "$@" | sed 's/^/### /'; echo "################################################################"; echo; }

# run_with_timeout SECONDS CMD... — coreutils timeout when available (Linux
# cron boxes), else a bash background-killer fallback (stock macOS has no
# timeout binary). Returns the command's exit code, or >128 when killed.
run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
    return $?
  fi
  "$@" &
  local cmd_pid=$!
  ( sleep "$secs"; kill "$cmd_pid" 2>/dev/null ) &
  local killer_pid=$!
  local rc
  wait "$cmd_pid"; rc=$?
  kill "$killer_pid" 2>/dev/null
  wait "$killer_pid" 2>/dev/null
  return "$rc"
}

# probe_milestone RUN POD MILESTONE ITER GAMES_AT REMOTE_CKPT LOCAL_NAME
probe_milestone() {
  local run="$1" pod="$2" milestone="$3" iter="$4" games_at="$5" remote_ckpt="$6" local_name="$7"
  local ledger_dir="$PROBE_ROOT/$run"
  local ledger="$ledger_dir/ledger.jsonl"
  local ckpt="$CURATED/$local_name"
  mkdir -p "$ledger_dir" "$CURATED"

  # -- fetch + byte-size verify the milestone checkpoint --------------------
  local remote_size
  remote_size="$("${KC[@]}" exec "$pod" -- sh -c "wc -c < '$remote_ckpt'" | tr -d '[:space:]')" || {
    log "  FAIL $run@$milestone: cannot stat remote checkpoint $remote_ckpt"; return 1; }
  if [ -f "$ckpt" ] && [ "$(wc -c < "$ckpt" | tr -d '[:space:]')" = "$remote_size" ]; then
    log "  checkpoint $local_name already present ($remote_size bytes) — reusing"
  else
    log "  kubectl cp $remote_ckpt -> $ckpt ($remote_size bytes, timeout ${CP_TIMEOUT}s)"
    run_with_timeout "$CP_TIMEOUT" "${KC_CP[@]}" cp "$pod:$remote_ckpt" "$ckpt" >/dev/null || {
      log "  FAIL $run@$milestone: kubectl cp failed or timed out; removing partial copy"
      rm -f "$ckpt"
      return 1
    }
    local local_size
    local_size="$(wc -c < "$ckpt" | tr -d '[:space:]')"
    if [ "$local_size" != "$remote_size" ]; then
      log "  FAIL $run@$milestone: size mismatch (local $local_size vs remote $remote_size); removing partial copy"
      rm -f "$ckpt"
      return 1
    fi
  fi

  # -- cross-pool Pearson (value-calibration, pool-self-v1 [+ pool-fp-v1]) --
  local pearson_self="$ledger_dir/pearson-pool-self-v1-$milestone.json"
  if [ ! -s "$pearson_self" ]; then
    log "  value-calibration on pool-self-v1 -> $pearson_self"
    uv run pokezero-neural value-calibration --checkpoint "$ckpt" \
      --data "$POOL_SELF" --json > "$pearson_self.tmp" \
      && mv "$pearson_self.tmp" "$pearson_self" || {
        rm -f "$pearson_self.tmp"
        log "  FAIL $run@$milestone: value-calibration (pool-self-v1) failed"; return 1; }
  fi
  local pearson_args=(--pearson "pool-self-v1=$pearson_self")
  if [ -f "$POOL_FP" ]; then
    local pearson_fp="$ledger_dir/pearson-pool-fp-v1-$milestone.json"
    if [ ! -s "$pearson_fp" ]; then
      log "  value-calibration on pool-fp-v1 -> $pearson_fp"
      uv run pokezero-neural value-calibration --checkpoint "$ckpt" \
        --data "$POOL_FP" --json > "$pearson_fp.tmp" \
        && mv "$pearson_fp.tmp" "$pearson_fp" || {
          rm -f "$pearson_fp.tmp"
          log "  FAIL $run@$milestone: value-calibration (pool-fp-v1) failed"; return 1; }
    fi
    pearson_args+=(--pearson "pool-fp-v1=$pearson_fp")
  fi

  # -- dV hazard probe (atomic out: tmp + rename, so a killed probe can't ---
  # -- leave a truncated non-empty JSON that poisons every later resume) ----
  local hazard_out="$ledger_dir/hazard-$milestone.json"
  local label="$run-i$iter"
  if [ ! -s "$hazard_out" ]; then
    log "  hazard probe ($HAZARD_GAMES games) -> $hazard_out"
    uv run python "$HAZARD_PROBE" \
      --checkpoint "$ckpt=$label" --showdown-root "$SHOWDOWN" \
      --games "$HAZARD_GAMES" --out "$hazard_out.tmp" \
      && mv "$hazard_out.tmp" "$hazard_out" || {
        rm -f "$hazard_out.tmp"
        log "  FAIL $run@$milestone: hazard probe failed"; return 1; }
  fi

  # -- one ledger line per milestone ----------------------------------------
  "${HELPER[@]}" record --ledger "$ledger" --run-id "$run" \
    --milestone "$milestone" --iteration "$iter" --games-at "$games_at" \
    --checkpoint "$ckpt" "${pearson_args[@]}" \
    --hazard "$hazard_out" --hazard-label "$label" || {
      log "  FAIL $run@$milestone: ledger record failed"; return 1; }
  return 0
}

# process_run POD — everything for one run; failures isolated by the caller.
process_run() {
  local pod="$1"

  local run
  run="$("${KC[@]}" get pod "$pod" -o jsonpath='{.spec.containers[0].args}' \
    | "${HELPER[@]}" run-id-from-args)" || {
    log "FAIL $pod: could not extract --run-id from pod args"; return 1; }
  log "run $run (pod $pod)"

  local run_root="$SHARED/$run"
  local status_json="$TMP/status-$run.json"
  "${KC[@]}" exec "$pod" -- cat "$run_root/STATUS.json" > "$status_json" || {
    log "FAIL $run: cannot read $run_root/STATUS.json"; return 1; }

  # -- plan: pending milestones vs the local ledger --------------------------
  local ledger="$PROBE_ROOT/$run/ledger.jsonl"
  local plan_tsv="$TMP/plan-$run.tsv"
  "${HELPER[@]}" plan --status-json "$status_json" --ledger "$ledger" \
    --step "$STEP" --max-distance "$MAX_DIST" --format tsv > "$plan_tsv" || {
    log "FAIL $run: milestone planning failed"; return 1; }
  grep '^#' "$plan_tsv" | sed 's/^# /  /'

  local failed=0
  local milestone iter games_at distance action remote_ckpt local_name skip_reason
  while IFS=$'\t' read -r milestone iter games_at distance action remote_ckpt local_name skip_reason; do
    [ -n "$milestone" ] || continue
    if [ "$action" = "skip" ]; then
      log "  milestone $milestone SKIPPED: $skip_reason"
      if [ "$DRY_RUN" -eq 1 ]; then
        log "  [dry-run] would record the SKIPPED milestone in the ledger"
      else
        "${HELPER[@]}" record --ledger "$ledger" --run-id "$run" \
          --milestone "$milestone" --iteration "$iter" --games-at "$games_at" \
          --checkpoint "$remote_ckpt" --skip-reason "$skip_reason" \
          || failed=$((failed + 1))
      fi
      continue
    fi
    log "  milestone ${milestone} -> iteration $iter (${games_at} games, distance ${distance}, $local_name)"
    if [ "$DRY_RUN" -eq 1 ]; then
      log "  [dry-run] would copy $remote_ckpt and run value-calibration + hazard probe"
      continue
    fi
    probe_milestone "$run" "$pod" "$milestone" "$iter" "$games_at" "$remote_ckpt" "$local_name" \
      || failed=$((failed + 1))
  done < <(grep -v '^#' "$plan_tsv")

  # -- ecology watchdogs over eval-timeline.jsonl -----------------------------
  local timeline="$TMP/timeline-$run.jsonl"
  local alarms="$TMP/alarms-$run.jsonl"
  # The drift baseline persists at first computation so later sweeps cannot
  # ratchet it forward as the tail window slides; dry-run reads but never writes.
  local wd_flags=(--run-id "$run" --timeline "$timeline"
                  --baseline-file "$PROBE_ROOT/$run/drift-baseline.json")
  [ "$DRY_RUN" -eq 0 ] || wd_flags+=(--no-persist)
  if "${KC[@]}" exec "$pod" -- sh -c "tail -n $TIMELINE_TAIL '$run_root/eval-timeline.jsonl'" > "$timeline"; then
    "${HELPER[@]}" watchdog "${wd_flags[@]}" > "$alarms" || {
      log "FAIL $run: watchdog evaluation failed"; return 1; }
    if [ -s "$alarms" ]; then
      loud "ECOLOGY ALARM — run $run" "$(cat "$alarms")"
      if [ "$DRY_RUN" -eq 0 ]; then
        mkdir -p "$PROBE_ROOT"
        cat "$alarms" >> "$ALERTS"
        log "alarms appended to $ALERTS"
      else
        log "[dry-run] alarms NOT appended to $ALERTS"
      fi
    else
      log "  watchdogs healthy"
    fi
  else
    log "FAIL $run: cannot read $run_root/eval-timeline.jsonl"; return 1
  fi

  [ "$failed" -eq 0 ] || { log "FAIL $run: $failed milestone(s) failed"; return 1; }
  return 0
}

# ---------------------------------------------------------------------------
log "sweep start (dry-run=$DRY_RUN, step=$STEP, kubectl timeout=$REQ_TIMEOUT)"

if [ "$DRY_RUN" -eq 0 ]; then
  # Pre-flights before taking the lock or touching the cluster.
  [ -f "$POOL_SELF" ] || {
    echo "[milestone-probes] FATAL: pool-self-v1 not found at $POOL_SELF — the cross-pool Pearson read needs the frozen belief-1.5m held-out pool (see docs/next_train_readiness_plan.md WS-3)" >&2
    exit 1
  }
  grep -q "value_self_hazard_response" "$HAZARD_PROBE" || {
    echo "[milestone-probes] FATAL: $HAZARD_PROBE has no dV section — this checkout predates the re-landed value-response hazard probe (#501); update main" >&2
    exit 1
  }
  # Exclusive sweep lock: fd 9 stays open for the life of this shell, so the
  # flock (taken by the python helper on the inherited fd — stock macOS has no
  # flock binary) is released on ANY exit, including kill -9.
  mkdir -p "$PROBE_ROOT"
  exec 9>"$LOCKFILE"
  if ! "${HELPER[@]}" lock --fd 9; then
    log "another sweep holds $LOCKFILE — exiting (no overlap allowed)"
    exit 0
  fi
fi

PODS="$("${KC[@]}" get pods --no-headers 2>/dev/null | awk '$3 == "Running" && $1 ~ /fnd-controller-/ {print $1}')" || {
  echo "[milestone-probes] FATAL: cannot list pods (check context/namespace env)" >&2; exit 1; }
if [ -z "$PODS" ]; then
  loud "NOTICE: no running fnd-controller pods — nothing probed, nothing watched" \
       "(if runs should be active, check POKEZERO_CLUSTER_CONTEXT/NAMESPACE and the pod naming)"
  exit 0
fi

FAILED_RUNS=0
TOTAL_RUNS=0
for pod in $PODS; do
  TOTAL_RUNS=$((TOTAL_RUNS + 1))
  process_run "$pod" || FAILED_RUNS=$((FAILED_RUNS + 1))
done

log "sweep done: $((TOTAL_RUNS - FAILED_RUNS))/$TOTAL_RUNS runs clean"
[ "$FAILED_RUNS" -eq 0 ] || exit 1
exit 0
