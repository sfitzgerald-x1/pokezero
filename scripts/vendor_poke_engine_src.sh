#!/usr/bin/env bash
# Vendor the poke-engine Rust source (gen3-patched) into third_party/poke-engine-src/
# for the native pokezero-search crate (rust/pokezero-search) to consume as a Cargo
# `path` dependency. Companion to scripts/setup_poke_engine.sh, which builds the
# *Python* binding from the same sdist + patches; this script vendors the *source*
# so Cargo can link the engine crate directly (no Python FFI in the search loop).
#
# The vendored tree is fetched, never committed: third_party/poke-engine-src/ is
# gitignored. Re-run this script after a clean checkout before building the crate.
#
# Patches applied (third_party/):
#   poke-engine-gen3-residual-order.patch — gen3 end-of-turn residual order fix
#   (see setup_poke_engine.sh header and docs/engine_fidelity_findings.md).
#   --fuzz=0 so a version bump fails loudly instead of applying hunks at
#   shifted locations.
#
# Requires: uv. Usage: scripts/vendor_poke_engine_src.sh [venv-python]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${1:-$REPO/.venv/bin/python}"
VERSION="0.0.47"
DEST="$REPO/third_party/poke-engine-src"
DL_DIR="$(mktemp -d -t poke-engine-src)"
trap 'rm -rf "$DL_DIR"' EXIT

echo "[1/3] fetch poke-engine==$VERSION sdist"
uv run --python "$PYTHON" pip download "poke-engine==$VERSION" --no-deps --no-binary :all: -d "$DL_DIR" >/dev/null
tar xzf "$DL_DIR"/poke_engine-"$VERSION".tar.gz -C "$DL_DIR"
SRC="$DL_DIR/poke_engine-$VERSION"

echo "[2/3] apply gen3 patches"
for patch in poke-engine-gen3-residual-order.patch; do
  (cd "$SRC" && patch -p1 --forward --fuzz=0 < "$REPO/third_party/$patch") && echo "      $patch: applied"
done

echo "[3/3] install into $DEST"
rm -rf "$DEST"
mv "$SRC" "$DEST"
echo "vendored poke-engine $VERSION (gen3-patched) at third_party/poke-engine-src/"
