#!/usr/bin/env bash
# Set up foul-play as an arms-length Gen 3 random-battle eval opponent.
#
# Builds an isolated venv with the canonical patched gen3 poke-engine and applies the local
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
for patch in foulplay-local-nosec.patch foulplay-rest-cant.patch foulplay-capture.patch; do
  if git -C "$FP" apply --reverse --check "$REPO/third_party/$patch" 2>/dev/null; then
    echo "      $patch: already applied"
  else
    git -C "$FP" apply "$REPO/third_party/$patch" && echo "      $patch: applied"
  fi
done

echo "[3/3] build isolated venv + poke-engine (gen3) via uv"
uv venv --python 3.12 "$FPVENV"
uv pip install --python "$FPVENV/bin/python" requests==2.33.0 websockets==14.1 python-dateutil==2.8.0
# Import success alone does not qualify this separate consumer: an unpatched
# gen3 wheel can panic on the exhausted ordinary-sleep fixture. Reuse the same
# pinned source and ordered patch manifest as the other engine builders.
bash "$REPO/scripts/setup_poke_engine.sh" "$FPVENV/bin/python"
"$FPVENV/bin/python" "$REPO/scripts/verify_foulplay_poke_engine.py" --iterations 128

echo "foul-play gen3 eval ready -> $FPVENV"
