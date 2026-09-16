"""Static contract checks for the model-enabled native build entrypoint."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_search_crate_model.sh"


def test_relative_interpreter_is_canonicalized_before_the_crate_directory_change() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")
    canonicalize = 'PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")"'
    assert canonicalize in text
    assert text.index(canonicalize) < text.index('cd "$REPO/rust/pokezero-search"')


def test_model_build_requires_the_tch_matched_torch_runtime() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "tch/torch-sys 0.26.0 requires PyTorch 2.13.0" in text
    assert "2.13.0|2.13.0+*" in text
