#!/usr/bin/env bash
# Option-A first GPU measurement (design note section 5) on one GPU: the census, the generated step's phase marks, warm
# walls (in-process CUPTI, then under nsys), JAX per phase (v2_engine_profile --save-trace --frames 40, then jax-attrib),
# ncu on the heaviest chunks and the hand kernels, the divergence estimate, the live pricer split and the report.
# Every path is an argument; nothing here names a cluster. Each step writes a fresh subdirectory of --out, prints its
# command before running it under `timeout`, and records its exit code in --out/status.tsv; a failed step does not stop
# the later ones (steps that need its output are skipped). Exit 0 only when every step ran and passed.
set -uo pipefail

usage() {
  cat <<'EOF'
usage: scripts/v2e_prof_a1.sh --config FILE --reference-run DIR --shards DIR --out DIR --build-dir DIR [options]

  --config FILE              run config (as gpu_engine.v2e_diff and v2_engine_profile take it)
  --reference-run DIR        the reference run (journal and initial actor)
  --shards DIR               capture shards/ directory (capture.json beside it)
  --out DIR                  output root; must not exist (one subdirectory per step)
  --build-dir DIR            generated CUDA build cache (a cached step .so there is reused)
  --listing FILE             pruned step_with_chance listing ("NNN: source"), for device lines per phase
  --python PY                interpreter (default: $PY, else python3)
  --d3-root DIR --chunk N    D3 bank chunk whose saved roots the live pricer split prices (step pricer; skipped without)
  --champion-archive FILE --champion-archive-sha256 HEX --champion-checkpoint-sha256 HEX
                             the champion (bf16) for the pricer split
  --compilation-cache DIR    absolute JAX persistent compilation cache for the pricer split
  --memory-policy            apply the bank runtime's JAX memory policy in the pricer split
  --live-metadata FILE --live-metadata-sha256 HEX
                             the live server's engine metadata: the pricer split compares its compiled tables with the bank's
  --live-tables-digest HEX   a recorded engine_tables_digest of the live executor's tables, for the same comparison
  --shard-limit N            marks: shards per lane count (default 20)
  --divergence-limit N       divergence: shards (default 100)
  --steps LIST               comma-separated subset of census,marks,walls,nsys,jax,ncu,divergence,pricer,report (default all)
  --dry-run                  print the commands without running them
EOF
}

CONFIG= REFERENCE= SHARDS= OUT= BUILD= LISTING= PYBIN=${PY:-python3} D3ROOT= CHUNK= ARCHIVE= ARCHIVE_SHA= CHECKPOINT_SHA= CACHE= MEMPOLICY=
LIVE_META= LIVE_META_SHA= LIVE_DIGEST=
SHARD_LIMIT=20 DIV_LIMIT=100 STEPS=census,marks,walls,nsys,jax,ncu,divergence,pricer,report DRY=0
while (($#)); do
  case "$1" in
    --config) CONFIG=$2; shift 2;;
    --reference-run) REFERENCE=$2; shift 2;;
    --shards) SHARDS=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --build-dir) BUILD=$2; shift 2;;
    --listing) LISTING=$2; shift 2;;
    --python) PYBIN=$2; shift 2;;
    --d3-root) D3ROOT=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --champion-archive) ARCHIVE=$2; shift 2;;
    --champion-archive-sha256) ARCHIVE_SHA=$2; shift 2;;
    --champion-checkpoint-sha256) CHECKPOINT_SHA=$2; shift 2;;
    --compilation-cache) CACHE=$2; shift 2;;
    --memory-policy) MEMPOLICY=--memory-policy; shift;;
    --live-metadata) LIVE_META=$2; shift 2;;
    --live-metadata-sha256) LIVE_META_SHA=$2; shift 2;;
    --live-tables-digest) LIVE_DIGEST=$2; shift 2;;
    --shard-limit) SHARD_LIMIT=$2; shift 2;;
    --divergence-limit) DIV_LIMIT=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
for name in CONFIG REFERENCE SHARDS OUT BUILD; do
  if [[ -z ${!name} ]]; then echo "missing required argument for $name" >&2; usage >&2; exit 2; fi
done

# Paths are used after `cd "$REPO"`: make the relative ones absolute first.
absolute() { case $1 in ''|/*) printf '%s' "$1";; *) printf '%s/%s' "$PWD" "$1";; esac; }
OUT=$(absolute "$OUT") CONFIG=$(absolute "$CONFIG") REFERENCE=$(absolute "$REFERENCE") SHARDS=$(absolute "$SHARDS") BUILD=$(absolute "$BUILD")
LISTING=$(absolute "$LISTING") D3ROOT=$(absolute "$D3ROOT") ARCHIVE=$(absolute "$ARCHIVE") CACHE=$(absolute "$CACHE") LIVE_META=$(absolute "$LIVE_META")
case $PYBIN in */*) PYBIN=$(absolute "$PYBIN");; esac
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export V2_SOURCE_COMMIT=${V2_SOURCE_COMMIT:-$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)}
cd "$REPO" || exit 2
if ((!DRY)); then
  if [[ -e $OUT ]]; then echo "$OUT exists; give a fresh output root" >&2; exit 2; fi
  mkdir -p "$(dirname "$OUT")" && mkdir "$OUT" || exit 2
  printf 'step\texit\tseconds\n' > "$OUT/status.tsv"
fi

PROFILE=(-m gpu_engine.v2e_phase_profile)
RUN=(--config "$CONFIG" --reference-run "$REFERENCE" --shards "$SHARDS" --build-dir "$BUILD")
FAILED=0
want() { [[ ",$STEPS," == *",$1,"* ]]; }
ok() { ((DRY)) || awk -F'\t' -v s="$1" '$1 == s && $2 == 0 {found = 1} END {exit !found}' "$OUT/status.tsv"; }

# run STEP SECONDS [ENV=VALUE ...] -- COMMAND...: print the command, run it under timeout, record the exit code.
run() {
  local step=$1 limit=$2; shift 2
  local env=()
  while [[ $1 != -- ]]; do env+=("$1"); shift; done; shift
  printf '\n=== %s (timeout %ss)\n+' "$step" "$limit"
  ((${#env[@]})) && printf ' %q' ${env[@]+"${env[@]}"}
  printf ' %q' timeout --kill-after=60 "$limit" "$@"; printf '\n'
  ((DRY)) && return 0
  local begin=$SECONDS code
  env ${env[@]+"${env[@]}"} timeout --kill-after=60 "$limit" "$@"; code=$?
  printf '%s\t%s\t%s\n' "$step" "$code" "$((SECONDS - begin))" >> "$OUT/status.tsv"
  ((code)) && { FAILED=1; echo "STEP_FAILED $step exit $code" >&2; }
  return 0
}

echo "repo $REPO commit $V2_SOURCE_COMMIT python $PYBIN out $OUT"

# Step 1: census (host-side tracing; tables load on the default platform, as v2e_diff loads them).
if want census; then
  run census 1800 -- "$PYBIN" -m gpu_engine.v2e_census --config "$CONFIG" --reference-run "$REFERENCE" --shards "$SHARDS" \
    ${LISTING:+--listing "$LISTING"} --out "$OUT/census"
fi

# Step 2: generated step. Marks (profile build), warm walls with per-chunk CUPTI times, then the same walls under nsys.
if want marks; then
  run marks 5400 -- "$PYBIN" "${PROFILE[@]}" marks "${RUN[@]}" --lanes 128 2048 --limit "$SHARD_LIMIT" --out "$OUT/marks"
fi
if want walls; then
  run walls 5400 -- "$PYBIN" "${PROFILE[@]}" walls "${RUN[@]}" --lanes 128 2048 8192 --calls 50 --out "$OUT/walls"
fi
if want nsys; then
  ((DRY)) || mkdir -p "$OUT/nsys"
  run nsys 3600 -- nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=false -o "$OUT/nsys/walls" \
    "$PYBIN" "${PROFILE[@]}" walls "${RUN[@]}" --lanes 128 2048 8192 --calls 50 --no-trace --no-jax --out "$OUT/nsys/walls-run"
  if ok nsys; then
    run nsys-stats 900 -- nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace --format csv --output "$OUT/nsys/walls" "$OUT/nsys/walls.nsys-rep"
  fi
fi

# Step 3: JAX per phase (frames kept to step_with_chance), then the offline attribution.
if want jax; then
  run jax-profile 7200 -- "$PYBIN" -B -u -m gpu_engine.v2_engine_profile --config "$CONFIG" --reference-run "$REFERENCE" --lanes 128 2048 \
    --frames 40 --save-trace --components --out "$OUT/jax-profile"
  if ok jax-profile; then
    run jax-attrib 1800 -- "$PYBIN" "${PROFILE[@]}" jax-attrib --profile-dir "$OUT/jax-profile" --lanes 128 2048 --out "$OUT/jax-attrib"
  fi
fi

# Step 4: ncu at 128 lanes on the heaviest chunk of the top two phases, chance_kernel and touch_kernel.
if want ncu; then
  if ok marks; then
    if ((DRY)); then REGEX='<ncu-targets output>'
    else REGEX=$("$PYBIN" "${PROFILE[@]}" ncu-targets --marks "$OUT/marks" --lanes 128 --top 2) || REGEX=; fi
    if [[ -n $REGEX ]]; then
      ((DRY)) || mkdir -p "$OUT/ncu"
      run ncu 3600 -- ncu --set full --target-processes all --kernel-name-base demangled --kernel-name "$REGEX" --launch-count 8 \
        -o "$OUT/ncu/step128" "$PYBIN" "${PROFILE[@]}" ncu-driver "${RUN[@]}" --lanes 128 --hand-kernels --marks "$OUT/marks" --out "$OUT/ncu/driver"
      if ok ncu; then
        run ncu-export 600 -- bash -c 'ncu --import "$1" --page details --csv > "$2" && ncu --import "$1" --page raw --csv > "$3"' _ \
          "$OUT/ncu/step128.ncu-rep" "$OUT/ncu/step128-details.csv" "$OUT/ncu/step128-raw.csv"
      fi
    else
      echo "ncu skipped: no targets from $OUT/marks" >&2; FAILED=1
    fi
  else
    echo "ncu skipped: marks did not pass" >&2; FAILED=1
  fi
fi

# Step 5: divergence estimate (host-side numpy over the shards).
if want divergence; then
  CENSUS_JSON=()
  ok census && CENSUS_JSON=(--census "$OUT/census/census.json")
  run divergence 3600 -- "$PYBIN" "${PROFILE[@]}" divergence "${RUN[@]}" --limit "$DIV_LIMIT" ${CENSUS_JSON[@]+"${CENSUS_JSON[@]}"} --out "$OUT/divergence"
fi

# Live pricer split: forward vs engine at the live shape (bucket 1, W=16), the d3_depth2_throughput --profile-split method.
if want pricer; then
  if [[ -n $D3ROOT ]]; then
    run pricer 7200 JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=0 -- "$PYBIN" "${PROFILE[@]}" pricer-split --d3-root "$D3ROOT" --chunk "$CHUNK" \
      --champion-archive "$ARCHIVE" --champion-archive-sha256 "$ARCHIVE_SHA" --champion-checkpoint-sha256 "$CHECKPOINT_SHA" \
      ${CACHE:+--compilation-cache "$CACHE"} ${MEMPOLICY:+"$MEMPOLICY"} ${LIVE_META:+--live-metadata "$LIVE_META"} \
      ${LIVE_META_SHA:+--live-metadata-sha256 "$LIVE_META_SHA"} ${LIVE_DIGEST:+--live-tables-digest "$LIVE_DIGEST"} --out "$OUT/pricer-split"
  else
    echo "pricer split skipped: no --d3-root" >&2
  fi
fi

# Output: phase_attribution.json, the table and the decision rule.
if want report; then
  REPORT_EXTRA=()
  if ((!DRY)); then
    found=$(ls "$OUT"/nsys/walls*cuda_gpu_trace*.csv 2>/dev/null | head -1)
    [[ -n $found ]] && REPORT_EXTRA=(--nsys-csv "$found")
    [[ -s $OUT/ncu/step128-details.csv ]] && REPORT_EXTRA+=(--ncu-details "$OUT/ncu/step128-details.csv" --ncu-raw "$OUT/ncu/step128-raw.csv")
  fi
  INPUTS=()
  for pair in census:census marks:marks walls:walls jax:jax-attrib divergence:divergence pricer:pricer-split; do
    [[ -d $OUT/${pair#*:} || $DRY == 1 ]] && INPUTS+=("--${pair%%:*}" "$OUT/${pair#*:}")
  done
  run report 600 -- "$PYBIN" "${PROFILE[@]}" report ${INPUTS[@]+"${INPUTS[@]}"} ${REPORT_EXTRA[@]+"${REPORT_EXTRA[@]}"} --out "$OUT/report"
fi

((DRY)) || { echo; cat "$OUT/status.tsv"; }
exit $FAILED
