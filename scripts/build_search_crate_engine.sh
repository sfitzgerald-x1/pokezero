#!/usr/bin/env bash
# Build and install the CPU-only pokezero-search extension used by engine
# differential tooling. This intentionally omits the optional Torch model
# feature and bounds Cargo parallelism for memory-safe certification rebuilds.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${1:-$(command -v python)}"
WHEEL_OUT="${2:-$REPO/rust/pokezero-search/target/wheels}"
CARGO_JOBS="${CARGO_BUILD_JOBS:-8}"

if [ ! -x "$PYTHON" ]; then
  echo "error: executable Python not found: $PYTHON" >&2
  exit 2
fi
case "$CARGO_JOBS" in
  ''|*[!0-9]*|0)
    echo "error: CARGO_BUILD_JOBS must be a positive integer" >&2
    exit 2
    ;;
esac

"$REPO/scripts/vendor_poke_engine_src.sh" "$PYTHON"
"$PYTHON" -m maturin --version >/dev/null
mkdir -p "$WHEEL_OUT"

(
  cd "$REPO/rust/pokezero-search"
  touch src/lib.rs
  CARGO_BUILD_JOBS="$CARGO_JOBS" "$PYTHON" -m maturin build \
    --release --skip-auditwheel --out "$WHEEL_OUT" -i "$PYTHON"
)

WHEEL="$(find "$WHEEL_OUT" -maxdepth 1 -name 'pokezero_search-*.whl' -print |
  LC_ALL=C sort | tail -1)"
if [ -z "$WHEEL" ]; then
  echo "error: pokezero-search wheel was not produced" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PYTHON" --force-reinstall "$WHEEL"
else
  "$PYTHON" -m pip install --force-reinstall "$WHEEL"
fi

PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$REPO/scripts/search_crate_branch_probe.py"
