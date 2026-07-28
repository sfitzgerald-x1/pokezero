#!/usr/bin/env bash
# Build poke-engine (gen3 feature) with pokezero's local gen3-correctness patches
# into a target venv. Used by the engine-swap work (docs/test_time_search_plan_v3.md);
# the fidelity differential (pokezero.engine_fidelity) is the patch's regression gate.
#
# Patches applied: third_party/poke-engine-gen3-patches.txt is the single source
# of truth for the list AND its order, shared with the other builder so the two
# can never drift (they did once — see that file's header). Per-patch rationale
# lives there and in docs/engine_fidelity_findings.md.
#
# Requires: rustup/cargo, uv. Usage: scripts/setup_poke_engine.sh [venv-python]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${1:-$REPO/.venv/bin/python}"
VERSION="0.0.47"
# Portable temp dir: `mktemp -d -t NAME` is BSD syntax; GNU coreutils rejects it
# ("too few X's in template"), which broke this script inside Linux images.
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/poke-engine-build.XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "[1/3] fetch poke-engine==$VERSION sdist"
uv run --python "$PYTHON" pip download "poke-engine==$VERSION" --no-deps --no-binary :all: -d "$BUILD_DIR" >/dev/null
tar xzf "$BUILD_DIR"/poke_engine-"$VERSION".tar.gz -C "$BUILD_DIR"
SRC="$BUILD_DIR/poke_engine-$VERSION"

echo "[2/3] apply gen3 patches"
PATCH_LIST="$REPO/third_party/poke-engine-gen3-patches.txt"
while IFS= read -r patch <&3; do
  case "$patch" in ''|'#'*) continue ;; esac
  if ! (cd "$SRC" && patch -p1 --forward --fuzz=0 < "$REPO/third_party/$patch"); then
    echo "ERROR: failed to apply $patch" >&2
    exit 1
  fi
  echo "      $patch: applied"
done 3< "$PATCH_LIST"

echo "[3/3] build + install (gen3 features) into $PYTHON"
uv pip install --python "$PYTHON" --no-cache --force-reinstall "$SRC" \
  --config-settings="build-args=--features poke-engine/gen3 --no-default-features"

"$PYTHON" -c "import poke_engine; print('patched poke-engine (gen3) ready')"
