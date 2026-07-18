#!/usr/bin/env bash
# Build poke-engine (gen3 feature) with pokezero's local gen3-correctness patches
# into a target venv. Used by the engine-swap work (docs/test_time_search_plan_v3.md);
# the fidelity differential (pokezero.engine_fidelity) is the patch's regression gate.
#
# Patches applied (third_party/):
#   poke-engine-gen3-residual-order.patch — end-of-turn residual order: items
#   (Leftovers, order 5) and Leech Seed (8) before poison/toxic (9) and burn (10)
#   damage, matching Gen 3 / Showdown. Upstream nets residuals against the
#   Leftovers heal (confirmed deviation, docs/engine_fidelity_findings.md).
#
# Requires: rustup/cargo, uv. Usage: scripts/setup_poke_engine.sh [venv-python]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${1:-$REPO/.venv/bin/python}"
VERSION="0.0.47"
BUILD_DIR="$(mktemp -d -t poke-engine-build)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "[1/3] fetch poke-engine==$VERSION sdist"
uv run --python "$PYTHON" pip download "poke-engine==$VERSION" --no-deps --no-binary :all: -d "$BUILD_DIR" >/dev/null
tar xzf "$BUILD_DIR"/poke_engine-"$VERSION".tar.gz -C "$BUILD_DIR"
SRC="$BUILD_DIR/poke_engine-$VERSION"

echo "[2/3] apply gen3 patches"
for patch in poke-engine-gen3-residual-order.patch; do
  (cd "$SRC" && patch -p1 --forward < "$REPO/third_party/$patch") && echo "      $patch: applied"
done

echo "[3/3] build + install (gen3 features) into $PYTHON"
uv pip install --python "$PYTHON" --no-cache --force-reinstall "$SRC" \
  --config-settings="build-args=--features poke-engine/gen3 --no-default-features"

"$PYTHON" -c "import poke_engine; print('patched poke-engine (gen3) ready')"
