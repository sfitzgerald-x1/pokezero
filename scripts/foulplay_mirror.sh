#!/usr/bin/env bash
# Generate foul-play vs foul-play (mirror) gen3 random battles on a local --no-security Showdown
# server, capturing each seat's protocol + /choose decisions for behavior-cloning teacher data.
#
# Both foul-play instances (GPL) run as SEPARATE processes in their own venv; pokezero (MIT)
# never imports foul-play. Each instance writes a capture transcript (received protocol blocks +
# outgoing /choose) via the FOULPLAY_CAPTURE_PATH hook. Run scripts/setup_foulplay_eval.sh first.
#
# Usage: POKEZERO_SHOWDOWN_ROOT=/path/to/pokemon-showdown \
#          scripts/foulplay_mirror.sh [games=10] [search_ms=150] [out_dir]
# Emits: <out_dir>/capA.jsonl and <out_dir>/capB.jsonl (one seat each).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FP="$REPO/third_party/foul-play"
FPVENV="${FOULPLAY_VENV:-$FP/.venv}"
SHOWDOWN="${POKEZERO_SHOWDOWN_ROOT:?set POKEZERO_SHOWDOWN_ROOT to a pokemon-showdown checkout}"
N="${1:-10}"; STM="${2:-150}"
OUT="${3:-$(mktemp -d /tmp/fpmirror.XXXXXX)}"; mkdir -p "$OUT"
WS="ws://localhost:8000/showdown/websocket"
cleanup(){ kill "${A:-}" "${B:-}" "${SRV:-}" 2>/dev/null; pkill -x node 2>/dev/null; }
trap cleanup EXIT

echo "[1] local Showdown server (--no-security)…"
( cd "$SHOWDOWN" && exec node pokemon-showdown start --no-security ) >"$OUT/server.log" 2>&1 & SRV=$!
for i in $(seq 1 90); do curl -sf -o /dev/null http://localhost:8000/ && break; sleep 1; done

run_fp(){ # username, capture_path, extra_args...
  local user="$1" cap="$2"; shift 2
  FOULPLAY_LOCAL_NOSEC=1 FOULPLAY_CAPTURE_PATH="$cap" PYTHONPATH="$FP" "$FPVENV/bin/python" "$FP/run.py" \
    --websocket-uri "$WS" --ps-username "$user" --pokemon-format gen3randombattle \
    --run-count "$N" --search-time-ms "$STM" "$@"
}

echo "[2] FoulPlayA (accept, ${STM}ms/move, $N games) -> $OUT/capA.jsonl"
run_fp FoulPlayA "$OUT/capA.jsonl" --bot-mode accept_challenge >"$OUT/foulplayA.log" 2>&1 & A=$!
sleep 10

echo "[3] FoulPlayB (challenge FoulPlayA, ${STM}ms/move, $N games) -> $OUT/capB.jsonl"
run_fp FoulPlayB "$OUT/capB.jsonl" --bot-mode challenge_user --user-to-challenge FoulPlayA \
  >"$OUT/foulplayB.log" 2>&1 & B=$!

echo "[4] playing $N mirror games…"
wait "$B" 2>/dev/null
sleep 3
echo "==================================================================="
echo "capture A: $(wc -l < "$OUT/capA.jsonl" 2>/dev/null || echo 0) lines"
echo "capture B: $(wc -l < "$OUT/capB.jsonl" 2>/dev/null || echo 0) lines"
echo "out dir:   $OUT"
echo "==================================================================="
