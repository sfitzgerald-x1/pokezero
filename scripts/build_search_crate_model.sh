#!/usr/bin/env bash
# Build + install the pokezero-search crate WITH in-crate TorchScript leaf
# evaluation (cargo feature `model`, tch-rs) against the venv's OWN libtorch.
#
# libtorch source policy: LIBTORCH_USE_PYTORCH=1 — the crate links the torch
# the venv already ships, never a vendored/downloaded libtorch, so Python-side
# and in-crate inference share one runtime. Version skew: tch 0.24.0
# (torch-sys) expects libtorch 2.11.0 while the venv ships torch 2.12.x, hence
# LIBTORCH_BYPASS_VERSION_CHECK=1. The build compiles torch-sys's C++ shims
# against the venv torch's real headers, and the machine-checkable
# compatibility proof is the parity gate:
#
#   python -m unittest tests.test_crate_model_leafeval
#
# (bit-exact crate-vs-Python outputs on the same TorchScript artifact). Bump
# tch and drop the bypass together when a torch-2.12-matched tch releases;
# re-run the parity gate after ANY tch or torch bump.
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
export LIBTORCH_BYPASS_VERSION_CHECK=1
export PYTHON
export PATH="$(dirname "$PYTHON"):$PATH"

echo "[1/4] torch in venv: $("$PYTHON" -c 'import torch; print(torch.__version__)')"

if [ ! -d "$REPO/third_party/poke-engine-src" ]; then
  echo "[2/4] vendoring poke-engine source"
  "$REPO/scripts/vendor_poke_engine_src.sh" "$PYTHON"
else
  echo "[2/4] poke-engine source already vendored"
fi

"$PYTHON" -m maturin --version >/dev/null 2>&1 || uv pip install --python "$PYTHON" maturin

echo "[3/4] maturin build --release --features model"
cd "$REPO/rust/pokezero-search"
"$PYTHON" -m maturin build --release --features model -i "$PYTHON"

echo "[4/4] install wheel"
WHEEL="$(ls -t target/wheels/pokezero_search-*.whl | head -1)"
uv pip install --python "$PYTHON" --force-reinstall "$WHEEL"
"$PYTHON" -c "import pokezero_search as m; assert m.MODEL_FEATURE_ENABLED, 'model feature missing from built wheel'; print('pokezero_search', m.__version__, '(model feature enabled)')"
