#!/usr/bin/env bash
# Milestone probe + ecology watchdog sweep over the active foundation runs
# (WS-3 items 2-3, docs/next_train_readiness_plan.md). Run manually or from a
# cron on an operator machine with kubectl access to the training cluster.
#
# Per invocation:
#   1. discover running *fnd-controller-* pods -> run ids (from pod --run-id arg)
#   2. per run, per ~100k-game milestone not yet in the local ledger:
#        - kubectl cp the milestone-nearest iteration checkpoint into
#          checkpoints/curated/<run>-i<iter>.pt (byte-size verified)
#        - cross-pool Pearson: pokezero-neural value-calibration on pool-self-v1
#          (and pool-fp-v1 when present)
#        - dV hazard probe: scripts/hazard_probe.py (the value-response variant;
#          fetched from git if the local copy predates the dV section)
#        - append one JSONL line to runs/milestone-probes/<run>/ledger.jsonl
#   3. ecology watchdogs over the run's eval-timeline.jsonl: game-length drift
#      (+50% vs the run's own 30k-100k band), matched-milestone max-damage
#      regression (-10 points vs run peak), policy_entropy < 0.35. Alarms
#      append to runs/milestone-probes/ALERTS.jsonl and print loudly.
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
#   POKEZERO_HAZARD_GAMES        hazard-probe corpus games          [default 150]
#   POKEZERO_MILESTONE_STEP      milestone spacing in games         [default 100000]
#   POKEZERO_TIMELINE_TAIL       eval-timeline rows fed to watchdogs [default 500]
#
# Usage (from repo root, `uv sync` done once):
#   scripts/milestone_probes.sh [--dry-run]
# --dry-run: read-only against the cluster — report runs discovered, pending
# milestones, and watchdog alarms; copy nothing, probe nothing, write nothing.

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
TIMELINE_TAIL="${POKEZERO_TIMELINE_TAIL:-500}"

KC=(kubectl --context "$CTX" -n "$NS")
HELPER=(python3 "$REPO/scripts/milestone_probes.py")
PROBE_ROOT="$REPO/runs/milestone-probes"
ALERTS="$PROBE_ROOT/ALERTS.jsonl"
CURATED="$REPO/checkpoints/curated"
POOL_SELF="$REPO/runs/e1-value-readiness-20260703/belief-1-5m/heldout-rollouts.jsonl"
POOL_FP="$REPO/runs/pool-fp-v1-20260704/pool-fp-v1.jsonl"
HAZARD_BRANCH="scott/hazard-probe-value-response"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/milestone-probes.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

log()   { echo "[milestone-probes] $*"; }
loud()  { echo; echo "################################################################"; printf '%s\n' "$@" | sed 's/^/### /'; echo "################################################################"; echo; }

# The dV hazard probe: prefer the checked-in script if it already has the value
# response section; otherwise pull the variant from git (local branch, then
# origin) and run that copy with PYTHONPATH pointed at this repo's src.
resolve_hazard_probe() {
  if grep -q "value_self_hazard_response" "$REPO/scripts/hazard_probe.py"; then
    echo "$REPO/scripts/hazard_probe.py"
    return 0
  fi
  local scratch="$TMP/hazard_probe_dv.py"
  if git -C "$REPO" show "$HAZARD_BRANCH:scripts/hazard_probe.py" > "$scratch" 2>/dev/null \
     || git -C "$REPO" show "origin/$HAZARD_BRANCH:scripts/hazard_probe.py" > "$scratch" 2>/dev/null; then
    echo "$scratch"
    return 0
  fi
  return 1
}

# probe_milestone RUN POD MILESTONE ITER REMOTE_CKPT LOCAL_NAME
probe_milestone() {
  local run="$1" pod="$2" milestone="$3" iter="$4" remote_ckpt="$5" local_name="$6"
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
    log "  kubectl cp $remote_ckpt -> $ckpt ($remote_size bytes)"
    "${KC[@]}" cp "$pod:$remote_ckpt" "$ckpt" >/dev/null || {
      log "  FAIL $run@$milestone: kubectl cp failed"; return 1; }
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

  # -- dV hazard probe -------------------------------------------------------
  local hazard_out="$ledger_dir/hazard-$milestone.json"
  local label="$run-i$iter"
  if [ ! -s "$hazard_out" ]; then
    local hazard_script
    hazard_script="$(resolve_hazard_probe)" || {
      log "  FAIL $run@$milestone: no dV-enabled hazard_probe.py (local or $HAZARD_BRANCH)"; return 1; }
    log "  hazard probe ($HAZARD_GAMES games) -> $hazard_out"
    PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" uv run python "$hazard_script" \
      --checkpoint "$ckpt=$label" --showdown-root "$SHOWDOWN" \
      --games "$HAZARD_GAMES" --out "$hazard_out" || {
        log "  FAIL $run@$milestone: hazard probe failed"; return 1; }
  fi

  # -- one ledger line per milestone ----------------------------------------
  "${HELPER[@]}" record --ledger "$ledger" --run-id "$run" \
    --milestone "$milestone" --iteration "$iter" --checkpoint "$ckpt" \
    "${pearson_args[@]}" --hazard "$hazard_out" --hazard-label "$label" || {
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
    --step "$STEP" --format tsv > "$plan_tsv" || {
    log "FAIL $run: milestone planning failed"; return 1; }
  grep '^#' "$plan_tsv" | sed 's/^# /  /'

  local failed=0
  local milestone iter games_at remote_ckpt local_name
  while IFS=$'\t' read -r milestone iter games_at remote_ckpt local_name; do
    [ -n "$milestone" ] || continue
    log "  milestone ${milestone} -> iteration $iter (${games_at} games, $local_name)"
    if [ "$DRY_RUN" -eq 1 ]; then
      log "  [dry-run] would copy $remote_ckpt and run value-calibration + hazard probe"
      continue
    fi
    probe_milestone "$run" "$pod" "$milestone" "$iter" "$remote_ckpt" "$local_name" \
      || failed=$((failed + 1))
  done < <(grep -v '^#' "$plan_tsv")

  # -- ecology watchdogs over eval-timeline.jsonl -----------------------------
  local timeline="$TMP/timeline-$run.jsonl"
  local alarms="$TMP/alarms-$run.jsonl"
  if "${KC[@]}" exec "$pod" -- sh -c "tail -n $TIMELINE_TAIL '$run_root/eval-timeline.jsonl'" > "$timeline"; then
    "${HELPER[@]}" watchdog --run-id "$run" --timeline "$timeline" > "$alarms" || {
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
log "sweep start (dry-run=$DRY_RUN, step=$STEP)"
PODS="$("${KC[@]}" get pods --no-headers 2>/dev/null | awk '$3 == "Running" && $1 ~ /fnd-controller-/ {print $1}')" || {
  echo "[milestone-probes] FATAL: cannot list pods (check context/namespace env)" >&2; exit 1; }
if [ -z "$PODS" ]; then
  log "no running fnd-controller pods — nothing to do"
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
