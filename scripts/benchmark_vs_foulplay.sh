#!/usr/bin/env bash
# Gen 3 random-battle eval: a PokeZero bot vs foul-play on a local --no-security Showdown
# server. foul-play (GPL) runs as a SEPARATE process in its own venv; pokezero never imports
# it (mere aggregation). Run scripts/setup_foulplay_eval.sh first.
#
# Usage: POKEZERO_SHOWDOWN_ROOT=/path/to/pokemon-showdown \
#          scripts/benchmark_vs_foulplay.sh <checkpoint.pt> [games=50] [search_ms=1000]
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FP="$REPO/third_party/foul-play"
FPVENV="${FOULPLAY_VENV:-$FP/.venv}"
PYBIN="${POKEZERO_PYTHON:-$REPO/.venv/bin/python}"
SHOWDOWN="${POKEZERO_SHOWDOWN_ROOT:?set POKEZERO_SHOWDOWN_ROOT to a pokemon-showdown checkout}"
CKPT="${1:?usage: $0 <checkpoint.pt> [games=50] [search_ms=1000]}"
N="${2:-50}"; STM="${3:-1000}"
PORT="${POKEZERO_BENCH_PORT:-8000}"   # override to run alongside another local server
WS="ws://localhost:${PORT}/showdown/websocket"
L="$(mktemp -d /tmp/fpbench.XXXXXX)"
# Kill only our own processes (NOT a broad `pkill node`) so a concurrent server on another port survives.
cleanup(){ kill "${FPID:-}" "${BOT:-}" "${SRV:-}" 2>/dev/null; }
trap cleanup EXIT

echo "[1] local Showdown server (--no-security) on :${PORT}…"
( cd "$SHOWDOWN" && exec node pokemon-showdown start "$PORT" --no-security ) >"$L/server.log" 2>&1 & SRV=$!
for i in $(seq 1 90); do curl -sf -o /dev/null "http://localhost:${PORT}/" && break; sleep 1; done

echo "[2] PokeZeroBot (accept, gen3randombattle, $N games)…"
"$PYBIN" "$REPO/scripts/play_online.py" --checkpoint "$CKPT" --showdown-root "$SHOWDOWN" \
  --websocket "$WS" --username PokeZeroBot --format gen3randombattle \
  --accept --no-login --max-games "$N" >"$L/bot.log" 2>&1 & BOT=$!
sleep 10

echo "[3] FoulPlayBot (challenge_user PokeZeroBot, ${STM}ms/move, run-count $N)…"
FOULPLAY_LOCAL_NOSEC=1 PYTHONPATH="$FP" "$FPVENV/bin/python" "$FP/run.py" \
  --websocket-uri "$WS" --ps-username FoulPlayBot --bot-mode challenge_user \
  --user-to-challenge PokeZeroBot --pokemon-format gen3randombattle \
  --run-count "$N" --search-time-ms "$STM" >"$L/foulplay.log" 2>&1 & FPID=$!

echo "[4] waiting for $N games (PokeZeroBot exits after --max-games)…"
wait "$BOT" 2>/dev/null
WINS=$(grep -c "we won!" "$L/bot.log" || true); WINS=${WINS:-0}
DONE=$(grep -cE "we won!|won\." "$L/bot.log" || true); DONE=${DONE:-0}
echo "==================================================================="
echo "RESULT: PokeZeroBot won ${WINS}/${DONE} vs FoulPlayBot (gen3randombattle)"
echo "==================================================================="
echo "(logs: $L)"
