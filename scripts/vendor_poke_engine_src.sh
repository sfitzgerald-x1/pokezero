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
# Patches applied: third_party/poke-engine-gen3-patches.txt is the single source
# of truth for the list AND its order, shared with the other builder so the two
# can never drift (they did once — see that file's header). Per-patch rationale
# lives there and in docs/engine_fidelity_findings.md.
#
# Requires: uv, rsync. Usage: scripts/vendor_poke_engine_src.sh [venv-python]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${1:-$REPO/.venv/bin/python}"
VERSION="0.0.47"
DEST="$REPO/third_party/poke-engine-src"
# Portable temp dir: `mktemp -d -t NAME` is BSD syntax that GNU coreutils rejects
# ("too few X's in template"), which broke vendoring inside the Linux image.
# An explicit template with trailing X's behaves identically on both.
DL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/poke-engine-src.XXXXXX")"
trap 'rm -rf "$DL_DIR"' EXIT

echo "[1/3] fetch poke-engine==$VERSION sdist"
uv run --python "$PYTHON" pip download "poke-engine==$VERSION" --no-deps --no-binary :all: -d "$DL_DIR" >/dev/null
ARCHIVE="$DL_DIR/poke_engine-$VERSION.tar.gz"
"$PYTHON" "$REPO/scripts/verify_poke_engine_source.py" \
  "$ARCHIVE" --expected-version "$VERSION"
tar xzf "$ARCHIVE" -C "$DL_DIR"
SRC="$DL_DIR/poke_engine-$VERSION"

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

# `patch` writes a <file>.orig backup whenever a hunk needs one, and those
# backups are pre-patch copies of files we just changed. Left in place they ride
# the rsync into the vendored tree and become a trap: a grep-based audit of
# src/gen3/*.rs can read the STALE copy and reach the opposite conclusion about
# what the engine does. Delete them at the source, so the `--delete` rsync below
# also clears any that a previous run already installed.
find "$SRC" -name '*.orig' -delete

echo "[3/3] install into $DEST"
# Keep the destination directory stable. Finder can recreate .DS_Store between
# rm and mv on macOS, making a clean vendor operation fail with ENOTEMPTY.
mkdir -p "$DEST"
rsync -a --delete --exclude='.DS_Store' "$SRC/" "$DEST/"
echo "vendored poke-engine $VERSION (gen3-patched) at third_party/poke-engine-src/"

# Do not stamp a vendored source tree. A valid certification stamp proves both
# the Python wheel and the native search crate were rebuilt from this tree; the
# coordinating two-consumer builder creates that stamp after installation.
