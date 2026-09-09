#!/usr/bin/env bash
# Build + install the pokezero-search crate WITH in-crate TorchScript leaf
# evaluation (cargo feature `model`, tch-rs) against the venv's OWN libtorch.
#
# libtorch source policy: LIBTORCH_USE_PYTORCH=1 — the crate links the torch
# the venv already ships, never a vendored/downloaded libtorch, so Python-side
# and in-crate inference share one runtime. tch/torch-sys 0.26.0 is matched
# to libtorch 2.13.0; do not use LIBTORCH_BYPASS_VERSION_CHECK to paper over
# an ABI or C++-API mismatch. The build rejects a different runtime before
# compiling torch-sys's C++ shims. The machine-checkable compatibility proof
# remains the parity gate:
#
#   python -m unittest tests.test_crate_model_leafeval
#
# (bit-exact crate-vs-Python outputs on the same TorchScript artifact). Bump
# tch and this explicit runtime guard together; re-run the parity gate after
# ANY tch or torch bump.
#
# The built extension embeds an rpath to the venv's torch/lib (build.rs), so
# `import pokezero_search` works without importing torch first.
#
# Requires: uv, cargo. Usage: scripts/build_search_crate_model.sh [venv-python]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${1:-$REPO/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "error: python not found at $PYTHON (pass the venv python as arg 1)" >&2
  exit 1
fi

export LIBTORCH_USE_PYTORCH=1
if [[ "${LIBTORCH_BYPASS_VERSION_CHECK+x}" == x ]]; then
  echo "error: LIBTORCH_BYPASS_VERSION_CHECK is forbidden; install the matching PyTorch 2.13.0 runtime" >&2
  exit 1
fi
export PYTHON
export PATH="$(dirname "$PYTHON"):$PATH"

if ! TORCH_VERSION="$("$PYTHON" -c 'import torch; print(torch.__version__)' 2>/dev/null)"; then
  echo "error: PyTorch is not importable from $PYTHON; tch/torch-sys 0.26.0 requires PyTorch 2.13.0" >&2
  exit 1
fi
case "$TORCH_VERSION" in
  2.13.0|2.13.0+*) ;;
  *)
    echo "error: tch/torch-sys 0.26.0 requires a PyTorch 2.13.0 runtime; found $TORCH_VERSION" >&2
    exit 1
    ;;
esac
echo "[1/4] torch in venv: $TORCH_VERSION (required 2.13.0)"

if [ ! -d "$REPO/third_party/poke-engine-src" ]; then
  echo "[2/4] vendoring poke-engine source"
  "$REPO/scripts/vendor_poke_engine_src.sh" "$PYTHON"
else
  echo "[2/4] poke-engine source already vendored"
fi

"$PYTHON" -m maturin --version >/dev/null 2>&1 || uv pip install --python "$PYTHON" maturin

echo "[3/4] maturin build --release --features model"
cd "$REPO/rust/pokezero-search"
# --skip-auditwheel: the extension deliberately LINKS the venv's own libtorch
# through an embedded rpath (build.rs). Letting maturin's auditwheel repair copy
# torch's .so files into the wheel produces a SECOND libtorch in the process, and
# both copies try to register the same dispatch fallbacks:
#   "Tried to register multiple backend fallbacks for the same dispatch key
#    Conjugate" -> abort on `import torch; import pokezero_search`.
# On Linux the repair only triggers once patchelf is present, so this stayed
# invisible until the crate was first built in a container.
"$PYTHON" -m maturin build --release --features model --skip-auditwheel -i "$PYTHON"

echo "[4/4] install wheel"
WHEEL="$(ls -t target/wheels/pokezero_search-*.whl | head -1)"
uv pip install --python "$PYTHON" --force-reinstall "$WHEEL"
"$PYTHON" -c "import pokezero_search as m; assert m.MODEL_FEATURE_ENABLED, 'model feature missing from built wheel'; print('pokezero_search', m.__version__, '(model feature enabled)')"
