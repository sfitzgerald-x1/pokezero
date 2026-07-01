#!/usr/bin/env bash
# Calibrate foul-play strength: foul-play (fast + strong search) vs the baseline ladder
# (random-legal, simple-legal, max-damage). All 6 matchups run concurrently on ONE local
# --no-security Showdown server (distinct usernames), each foul-play challenging its paired
# baseline bot. Reports foul-play's win rate per matchup. foul-play (GPL) runs as separate
# processes; pokezero never imports it. Run scripts/setup_foulplay_eval.sh first.
#
# Usage: POKEZERO_SHOWDOWN_ROOT=/path scripts/benchmark_foulplay_baselines.sh [games=100] [fast_ms=150] [strong_ms=1000]
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FP="${FOULPLAY_DIR:-$REPO/third_party/foul-play}"
FPVENV="${FOULPLAY_VENV:-$FP/.venv}"
PY="${POKEZERO_PYTHON:-$REPO/.venv/bin/python}"
SHOWDOWN="${POKEZERO_SHOWDOWN_ROOT:?set POKEZERO_SHOWDOWN_ROOT}"
GAMES="${1:-100}"; FAST_MS="${2:-150}"; STRONG_MS="${3:-1000}"
OUT="${OUT_DIR:-$(mktemp -d /tmp/fpcal.XXXXXX)}"; mkdir -p "$OUT"
export SHOWDOWN_MAX_TURNS="${SHOWDOWN_MAX_TURNS:-250}"  # cap stall-war games -> tie at 250 turns
WS="ws://localhost:8000/showdown/websocket"
cleanup(){ kill $(jobs -p) 2>/dev/null; pkill -x node 2>/dev/null; pkill -f play_online_baseline 2>/dev/null; pkill -f "$FP/run.py" 2>/dev/null; }
trap cleanup EXIT

# tag  baseline       search_ms
COMBOS=(
  "fr random-legal $FAST_MS"
  "fs simple-legal $FAST_MS"
  "fm max-damage   $FAST_MS"
  "sr random-legal $STRONG_MS"
  "ss simple-legal $STRONG_MS"
  "sm max-damage   $STRONG_MS"
)

echo "[1] local Showdown server (--no-security)…"
( cd "$SHOWDOWN" && exec node pokemon-showdown start --no-security ) >"$OUT/server.log" 2>&1 &
for i in $(seq 1 90); do curl -sf -o /dev/null http://localhost:8000/ && break; sleep 1; done

echo "[2] starting 6 baseline bots (accept, $GAMES games each)…"
BOTPIDS=()
for c in "${COMBOS[@]}"; do set -- $c; tag=$1; base=$2
  "$PY" "$REPO/scripts/play_online_baseline.py" --policy "$base" --showdown-root "$SHOWDOWN" \
    --websocket "$WS" --username "b$tag" --format gen3randombattle --accept --no-login \
    --max-games "$GAMES" >"$OUT/bot-$tag.log" 2>&1 &
  BOTPIDS+=($!)
done
# Extra matchup: foul-play(fast) accepts, foul-play(strong) challenges -> search-strength gap.
# Names must NOT collide with any combo name (b$tag/f$tag), so use fpfast/fpstrong.
FOULPLAY_LOCAL_NOSEC=1 PYTHONPATH="$FP" "$FPVENV/bin/python" "$FP/run.py" \
  --websocket-uri "$WS" --ps-username fpfast --bot-mode accept_challenge \
  --pokemon-format gen3randombattle --run-count "$GAMES" --search-time-ms "$FAST_MS" \
  >"$OUT/fp-fpfast.log" 2>&1 &
FFPID=$!
sleep 12

echo "[3] starting 6 foul-play challengers + strong-vs-fast challenger…"
for c in "${COMBOS[@]}"; do set -- $c; tag=$1; ms=$3
  FOULPLAY_LOCAL_NOSEC=1 PYTHONPATH="$FP" "$FPVENV/bin/python" "$FP/run.py" \
    --websocket-uri "$WS" --ps-username "f$tag" --bot-mode challenge_user --user-to-challenge "b$tag" \
    --pokemon-format gen3randombattle --run-count "$GAMES" --search-time-ms "$ms" \
    >"$OUT/fp-$tag.log" 2>&1 &
done
FOULPLAY_LOCAL_NOSEC=1 PYTHONPATH="$FP" "$FPVENV/bin/python" "$FP/run.py" \
  --websocket-uri "$WS" --ps-username fpstrong --bot-mode challenge_user --user-to-challenge fpfast \
  --pokemon-format gen3randombattle --run-count "$GAMES" --search-time-ms "$STRONG_MS" \
  >"$OUT/fp-fpstrong.log" 2>&1 &

echo "[4] playing (waiting for the 6 baseline bots + fast accepter to finish $GAMES games each)…"
wait "${BOTPIDS[@]}" "$FFPID" 2>/dev/null

echo ""
echo "================= foul-play vs baselines ($GAMES games each) ================="
printf "%-14s %-14s %8s %16s %6s\n" "foul-play" "baseline" "search" "foul-play win%" "ties"
for c in "${COMBOS[@]}"; do set -- $c; tag=$1; base=$2; ms=$3
  log="$OUT/bot-$tag.log"
  # bot log: "we won!" = baseline won; "<foulplay> won." = foul-play won; "ended in a tie" = tie.
  bwins=$(grep -c "we won!" "$log" 2>/dev/null); bwins=${bwins:-0}
  fpwins=$(grep -c "won\." "$log" 2>/dev/null); fpwins=${fpwins:-0}
  ties=$(grep -c "ended in a tie" "$log" 2>/dev/null); ties=${ties:-0}
  total=$(( bwins + fpwins + ties ))
  pct=$([ "$total" -gt 0 ] && awk "BEGIN{printf \"%.0f\", 100*$fpwins/$total}" || echo "-")
  printf "%-14s %-14s %7sms %12s (%d/%d) %5d\n" "$tag" "$base" "$ms" "${pct}%" "$fpwins" "$total" "$ties"
done
# strong-vs-fast foul-play head-to-head (foul-play's own "Winner: <name>" log; None = tie).
total=$(grep -c "Winner:" "$OUT/fp-fpfast.log" 2>/dev/null); total=${total:-0}
swins=$(grep -ci "Winner: fpstrong" "$OUT/fp-fpfast.log" 2>/dev/null); swins=${swins:-0}
sties=$(grep -c "Winner: None" "$OUT/fp-fpfast.log" 2>/dev/null); sties=${sties:-0}
spct=$([ "$total" -gt 0 ] && awk "BEGIN{printf \"%.0f\", 100*$swins/$total}" || echo "-")
echo "-----------------------------------------------------------------------------"
printf "foul-play STRONG (%sms) vs FAST (%sms): strong win%% = %s%% (%d/%d), ties=%d\n" \
  "$STRONG_MS" "$FAST_MS" "${spct}" "$swins" "$total" "$sties"
echo "(logs: $OUT)"
