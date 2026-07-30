#!/usr/bin/env bash
# Rebuild BOTH consumers of the patched engine for certification:
#   1. the Python poke_engine wheel; 2. the native pokezero_search crate.
# A fingerprint stamp is allowed only after both installs succeed. Building one
# consumer and stamping it made stale mixed-engine measurements look current.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${1:-$(command -v python)}"
WHEEL_PARENT="${2:-$REPO/rust/pokezero-search/target/wheels}"
CARGO_JOBS="${CARGO_BUILD_JOBS:-8}"

if [ ! -x "$PYTHON" ]; then
  echo "error: executable Python not found: $PYTHON" >&2
  exit 2
fi
PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")"
case "$CARGO_JOBS" in
  ''|*[!0-9]*|0)
    echo "error: CARGO_BUILD_JOBS must be a positive integer" >&2
    exit 2
    ;;
esac

mkdir -p "$WHEEL_PARENT"
WHEEL_PARENT="$(cd "$WHEEL_PARENT" && pwd)"
WHEEL_OUT="$(mktemp -d "$WHEEL_PARENT/cert-wheel.XXXXXX")"
trap 'rm -rf "$WHEEL_OUT"' EXIT

echo "[1/8] vendor patched poke-engine source"
"$REPO/scripts/vendor_poke_engine_src.sh" "$PYTHON"

# Cargo considers timestamps when deciding whether a path dependency changed.
# A vendor rsync can preserve old mtimes, so force every vendored Rust input
# newer before compiling the crate.
echo "[2/8] invalidate Cargo source mtimes"
find "$REPO/third_party/poke-engine-src" -type f \( -name '*.rs' -o -name Cargo.toml -o -name Cargo.lock \) -exec touch {} +
find "$REPO/rust/pokezero-search" -type f \( -name '*.rs' -o -name Cargo.toml -o -name Cargo.lock \) -exec touch {} +

echo "[3/8] rebuild and install Python poke_engine"
"$REPO/scripts/setup_poke_engine.sh" "$PYTHON"

echo "[4/8] build fresh pokezero_search wheel"
"$PYTHON" -m maturin --version >/dev/null
(
  cd "$REPO/rust/pokezero-search"
  CARGO_BUILD_JOBS="$CARGO_JOBS" "$PYTHON" -m maturin build \
    --release --skip-auditwheel --out "$WHEEL_OUT" -i "$PYTHON"
)

WHEELS=("$WHEEL_OUT"/pokezero_search-*.whl)
if [ "${#WHEELS[@]}" -ne 1 ] || [ ! -f "${WHEELS[0]}" ]; then
  echo "error: expected exactly one fresh pokezero_search wheel in $WHEEL_OUT" >&2
  exit 1
fi
WHEEL="${WHEELS[0]}"

echo "[5/8] install exactly the fresh wheel"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PYTHON" --force-reinstall "$WHEEL"
else
  "$PYTHON" -m pip install --force-reinstall "$WHEEL"
fi

echo "[6/8] attest the two installed consumers"
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PY'
import poke_engine
import pokezero_search
from pathlib import Path

for module in (poke_engine, pokezero_search):
    path = Path(module.__file__ or "").resolve()
    if not path.is_file():
        raise SystemExit(f"installed module has no file: {module.__name__}")
    print(f"{module.__name__}: {path}")
PY
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$REPO/scripts/engine_build_fingerprint.py" \
  --write --after-two-consumer-rebuild
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$REPO/scripts/engine_build_fingerprint.py" --check

echo "[7/8] run behavioral engine probes"
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$REPO/scripts/engine_behavioral_probes.py"

echo "[8/8] verify branch-events mapper against the installed crate"
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$REPO/scripts/search_crate_branch_probe.py"
