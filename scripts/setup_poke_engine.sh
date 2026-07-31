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
# Requires: rustup/cargo, uv, git. Usage: scripts/setup_poke_engine.sh [venv-python]
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
ARCHIVE="$BUILD_DIR/poke_engine-$VERSION.tar.gz"
"$PYTHON" "$REPO/scripts/verify_poke_engine_source.py" \
  "$ARCHIVE" --expected-version "$VERSION"
tar xzf "$ARCHIVE" -C "$BUILD_DIR"
SRC="$BUILD_DIR/poke_engine-$VERSION"

echo "[2/3] apply gen3 patches"
"$PYTHON" "$REPO/scripts/apply_poke_engine_patches.py" "$SRC"

# Installed from the sdist ROOT, never from poke-engine-py/. The root pyproject
# sets `python-source = "python"` and `module-name = "poke_engine.poke_engine"`,
# which is what ships poke-engine-py's PYTHON half alongside the compiled
# extension. That half defines `monte_carlo_tree_search`, the entrypoint
# pokezero.engine_search calls; an install carrying only the extension imports
# cleanly, exposes the native `mcts`, and then dies mid-search on a bare
# AttributeError. tests/test_engine_search_no_panic.py probes for it and skips
# with this command rather than failing.
echo "[3/3] build + install (gen3 features) into $PYTHON"
uv pip install --python "$PYTHON" --no-cache --force-reinstall "$SRC" \
  --config-settings="build-args=--features poke-engine/gen3 --no-default-features"

"$PYTHON" -c "import poke_engine; print('patched poke-engine (gen3) ready')"

# Do not stamp here. Certification evidence covers two consumers of this patch
# set: this Python wheel and rust/pokezero-search. The coordinating rebuild
# script stamps only after both are rebuilt and installed.
