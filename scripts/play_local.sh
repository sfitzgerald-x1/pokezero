#!/usr/bin/env bash
# Play PokeZero checkpoints in a browser, FULLY LOCAL — nothing touches play.pokemonshowdown.com.
#
# It starts three things and ties them together:
#   1. a local Pokemon Showdown *sim server* (--no-security) on :8000  (websocket + battle engine)
#   2. one bot per checkpoint (PokeZeroBot-<name>) that accepts gen3randombattle challenges
#   3. a local Pokemon Showdown *web client* on :8080, pointed at the local sim
#
# Why both servers: the sim checkout is server-only — bare http://localhost:8000 just hangs at
# "Loading client..." because it tries to load the client from the production CDN. So we serve the
# real client locally on :8080 and connect it to the local sim via testclient-new.html?~~localhost:8000.
#
# Requires: node, python3, a local pokemon-showdown (sim) + pokemon-showdown-client checkout, and
# the repo .venv (pip install -e '.[neural]').
#   POKEZERO_SHOWDOWN_ROOT    path to the pokemon-showdown (sim) checkout            [required]
#   POKEZERO_SHOWDOWN_CLIENT  path to the pokemon-showdown-client checkout           [default: <root>/../pokemon-showdown-client]
#   POKEZERO_PYTHON           python to run the bot                                  [default: .venv/bin/python]
#
# Usage:
#   POKEZERO_SHOWDOWN_ROOT=/path/to/pokemon-showdown scripts/play_local.sh
#       -> serves the committed milestones as PokeZeroBot-500k and PokeZeroBot-1M
#   scripts/play_local.sh 500k=checkpoints/pokezero-gen3-500k.pt foo=runs/.../transformer-policy.pt
#       -> one PokeZeroBot-<name> per name=checkpoint argument
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${POKEZERO_PYTHON:-$REPO/.venv/bin/python}"
SIM="${POKEZERO_SHOWDOWN_ROOT:?set POKEZERO_SHOWDOWN_ROOT to a pokemon-showdown (sim) checkout}"
CLIENT="${POKEZERO_SHOWDOWN_CLIENT:-$(dirname "$SIM")/pokemon-showdown-client}"
CLIENT_DIR="$CLIENT/play.pokemonshowdown.com"
SIM_PORT="${POKEZERO_SIM_PORT:-8000}"; CLIENT_PORT="${POKEZERO_CLIENT_PORT:-8080}"
WS="ws://localhost:${SIM_PORT}/showdown/websocket"
L="$(mktemp -d /tmp/pokezero-play.XXXXXX)"

[ -d "$CLIENT_DIR" ] || { echo "client not found at $CLIENT_DIR — set POKEZERO_SHOWDOWN_CLIENT"; exit 1; }

VARIANTS=("$@")
[ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=("500k=$REPO/checkpoints/pokezero-gen3-500k.pt" "1M=$REPO/checkpoints/pokezero-gen3-1m.pt")

pids=()
cleanup(){ kill "${pids[@]}" 2>/dev/null; pkill -x node 2>/dev/null; }
trap cleanup EXIT INT TERM
pkill -x node 2>/dev/null; pkill -f play_online 2>/dev/null

echo "[1/3] sim server on :${SIM_PORT} (--no-security)…"
( cd "$SIM" && exec node pokemon-showdown start --no-security ) >"$L/server.log" 2>&1 & pids+=($!)
for i in $(seq 1 120); do curl -sf -o /dev/null "http://localhost:${SIM_PORT}/" && break; sleep 1; done

echo "[2/3] bots…"
for v in "${VARIANTS[@]}"; do
  name="${v%%=*}"; ckpt="${v#*=}"
  echo "        PokeZeroBot-${name}  <-  ${ckpt}"
  "$PY" "$REPO/scripts/play_online.py" --checkpoint "$ckpt" --showdown-root "$SIM" \
    --websocket "$WS" --username "PokeZeroBot-${name}" --format gen3randombattle \
    --accept --no-login --max-games 0 >"$L/bot-${name}.log" 2>&1 & pids+=($!)
done

echo "[3/3] local web client on :${CLIENT_PORT}…"
cat > "$CLIENT_DIR/index.html" <<HTML
<!doctype html><meta charset="utf-8"><title>PokeZero — local</title>
<script>location.replace("testclient-new.html?~~localhost:${SIM_PORT}");</script>
<a href="testclient-new.html?~~localhost:${SIM_PORT}">Loading the PokeZero local client…</a>
HTML
( cd "$CLIENT_DIR" && exec python3 -m http.server "$CLIENT_PORT" ) >"$L/client.log" 2>&1 & pids+=($!)

cat <<EOF

======================================================================
 READY  ->  open  http://localhost:${CLIENT_PORT}
 Pick any name, then challenge to [Gen 3] Random Battle:
$(for v in "${VARIANTS[@]}"; do echo "   - PokeZeroBot-${v%%=*}"; done)
 Fully local (client :${CLIENT_PORT} + sim :${SIM_PORT}); logs in ${L}
 Ctrl-C to stop everything.
======================================================================
EOF
wait
