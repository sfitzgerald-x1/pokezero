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
# Patches applied (third_party/), IN ORDER:
#   poke-engine-gen3-residual-order.patch — gen3 end-of-turn residual order fix
#   (see setup_poke_engine.sh header and docs/engine_fidelity_findings.md).
#   poke-engine-gen3-attract.patch — gen3 Attract 50% move-immobilization fix
#   (see setup_poke_engine.sh header and docs/engine_fidelity_findings.md);
#   authored against the residual-patched tree, applied AFTER residual-order.
#   poke-engine-gen3-struggle-typeless.patch — gen3 Struggle is TYPELESS (neutral
#   vs all types incl. Ghost, no STAB), per real Gen II+ mechanics; upstream
#   defines Struggle as Normal-type, which wrongly made gen3 Struggle immune vs
#   Ghost and resisted by Rock/Steel. Compile-time gated so gen1 stays Normal;
#   authored against the attract-patched tree, applied AFTER attract.
#   poke-engine-gen3-rapidspin-fidelity.patch — gen3 Rapid Spin / Protect fidelity.
#   (1) Protect leak: the move-id-keyed post-hit handlers (choice_hazard_clear,
#   choice_special_effect) survived remove_effects_for_protect() (which leaves
#   move_id intact), so a Protect-blocked Rapid Spin still stripped the spinner's
#   own Spikes, and Seismic Toss/Super Fang/Endeavor etc. still dealt fixed damage
#   THROUGH Protect. before_move now returns a blocked_by_protect bool threaded to
#   those handlers, which early-return when blocked (never gated on damage/hit_sub,
#   so a spin that connects on a Substitute STILL clears). (2) A connecting Rapid
#   Spin now also frees the user from Leech Seed and partial-trapping (Wrap/Bind/
#   Fire Spin/Clamp/Whirlpool) — previously unmodelled. Verified vs real gen3
#   Showdown (scripts/rapidspin_differential.py). Touches generate_instructions.rs
#   + choice_effects.rs only (no overlap with the other patches); authored against
#   the struggle-patched tree, applied AFTER struggle-typeless.
#   poke-engine-gen3-ability-fidelity.patch — full Gen 3 randbats ability audit
#   corrections. Includes exact chance branches and status/volatile immunity,
#   recoil/Wonder Guard/Sturdy, Forecast/weather suppression, gender-aware Cute
#   Charm, Intimidate through Substitute, Gen 3 Lightning Rod singles semantics,
#   and speed-tie isolation.
#   Authored against the rapidspin-patched tree and applied last.
#   --fuzz=0 so a version bump fails loudly instead of applying hunks at
#   shifted locations.
#
# Requires: uv, rsync. Usage: scripts/vendor_poke_engine_src.sh [venv-python]
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
for patch in \
  poke-engine-gen3-residual-order.patch \
  poke-engine-gen3-attract.patch \
  poke-engine-gen3-struggle-typeless.patch \
  poke-engine-gen3-rapidspin-fidelity.patch \
  poke-engine-gen3-ability-fidelity.patch; do
  if ! (cd "$SRC" && patch -p1 --forward --fuzz=0 < "$REPO/third_party/$patch"); then
    echo "ERROR: failed to apply $patch" >&2
    exit 1
  fi
  echo "      $patch: applied"
done

echo "[3/3] install into $DEST"
# Keep the destination directory stable. Finder can recreate .DS_Store between
# rm and mv on macOS, making a clean vendor operation fail with ENOTEMPTY.
mkdir -p "$DEST"
rsync -a --delete --exclude='.DS_Store' "$SRC/" "$DEST/"
echo "vendored poke-engine $VERSION (gen3-patched) at third_party/poke-engine-src/"
