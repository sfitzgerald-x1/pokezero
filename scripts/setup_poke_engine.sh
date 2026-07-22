#!/usr/bin/env bash
# Build poke-engine (gen3 feature) with pokezero's local gen3-correctness patches
# into a target venv. Used by the engine-swap work (docs/test_time_search_plan_v3.md);
# the fidelity differential (pokezero.engine_fidelity) is the patch's regression gate.
#
# Patches applied (third_party/), IN ORDER:
#   poke-engine-gen3-residual-order.patch — end-of-turn residual order split to
#   match Showdown gen3: Leftovers (5) + Shed Skin (5.3), then Leech Seed (8),
#   then poison/toxic/burn damage (9/10), then threshold berries + Rain Dish +
#   Speed Boost (10+). Upstream ran ALL items/abilities after status damage,
#   netting residuals against the Leftovers heal (confirmed deviation,
#   docs/engine_fidelity_findings.md).
#   poke-engine-gen3-attract.patch — Attract (infatuation) 50%-per-turn move
#   immobilization. Upstream accepts the ATTRACT volatile but wholly ignores it
#   (zero behavioral references in gen3); the patch adds the 50/50 chance branch
#   mirroring the confusion self-hit, so search prices the immobilization
#   (docs/engine_fidelity_findings.md). Authored against the residual-patched
#   tree, so it is applied AFTER residual-order.
#   poke-engine-gen3-struggle-typeless.patch — gen3 Struggle is TYPELESS (neutral
#   vs all types incl. Ghost, no STAB); compile-time gated so gen1 stays Normal.
#   Applied AFTER attract. (Kept in lock-step with vendor_poke_engine_src.sh so the
#   Python wheel and the native pokezero-search crate build the SAME gen3 engine.)
#   poke-engine-gen3-rapidspin-fidelity.patch — gen3 Rapid Spin / Protect fidelity:
#   Protect-blocked move-id-keyed handlers (hazard clear, Seismic Toss/Super Fang/
#   Endeavor special effects) no longer fire through Protect, and a connecting
#   Rapid Spin now also clears the user's Leech Seed + partial-trap. Regression
#   gate: scripts/rapidspin_differential.py + tests/test_engine_rapidspin_fidelity.py.
#   Applied AFTER struggle-typeless.
#   poke-engine-gen3-ability-fidelity.patch — randbats ability audit fixes:
#   recoil/Struggle/Wonder Guard/Sturdy, exact contact and Shed Skin chance
#   branches, status and volatile immunities, Synchronize/Early Bird, Sand Veil,
#   Forecast/weather suppression, gender-aware Cute Charm, Gen 3 Lightning Rod
#   singles semantics, and speed-tie Choice isolation. Applied AFTER
#   rapidspin-fidelity.
#   --fuzz=0 so a future version bump fails loudly instead of applying hunks at
#   shifted locations.
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

echo "[3/3] build + install (gen3 features) into $PYTHON"
uv pip install --python "$PYTHON" --no-cache --force-reinstall "$SRC" \
  --config-settings="build-args=--features poke-engine/gen3 --no-default-features"

"$PYTHON" -c "import poke_engine; print('patched poke-engine (gen3) ready')"
