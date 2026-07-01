#!/usr/bin/env bash
# Set up foul-play as an arms-length Gen 3 random-battle eval opponent.
#
# Builds an isolated venv with poke-engine compiled for gen3 and applies the local
# --no-security login patch. foul-play is GPL-3.0 and runs as a SEPARATE process; pokezero
# (MIT) never imports it. Requires: rustup/cargo, uv. See third_party/README.md.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FP="$REPO/third_party/foul-play"
FPVENV="${FOULPLAY_VENV:-$FP/.venv}"

echo "[1/3] materialize submodule"
git -C "$REPO" submodule update --init "$FP"

echo "[2/3] apply foul-play patches (idempotent)"
# --no-security login patch (local eval servers) + Rest 'cant' crash fix (rest_turns==1 called
# exit(1), killing long gen3 games with Rest/Sleep Talk; the fix decrements to 0 and continues).
for patch in foulplay-local-nosec.patch foulplay-rest-cant.patch; do
  if git -C "$FP" apply --reverse --check "$REPO/third_party/$patch" 2>/dev/null; then
    echo "      $patch: already applied"
  else
    git -C "$FP" apply "$REPO/third_party/$patch" && echo "      $patch: applied"
  fi
done

echo "[3/3] build isolated venv + poke-engine (gen3) via uv"
uv venv --python 3.12 "$FPVENV"
uv pip install --python "$FPVENV/bin/python" requests==2.33.0 websockets==14.1 python-dateutil==2.8.0
uv pip install --python "$FPVENV/bin/python" --no-cache poke-engine==0.0.47 \
  --config-settings="build-args=--features poke-engine/gen3 --no-default-features"

"$FPVENV/bin/python" -c "import poke_engine; print('foul-play gen3 eval ready ->', '$FPVENV')"
