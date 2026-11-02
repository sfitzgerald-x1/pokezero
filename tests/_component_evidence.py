"""Scratch paths for existing component-test captures, with optional preservation."""
from pathlib import Path
import os
import tempfile


def capture_path(case, name):
    if Path(name).name != name:
        raise ValueError('component capture requires a filename')
    preserved = os.environ.get('POKEZERO_TEST_EVIDENCE_DIR')
    if preserved:
        Path(preserved).mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix='pokezero-component-', dir=preserved))
    else:
        scratch = tempfile.TemporaryDirectory(prefix='pokezero-component-')
        cleanup = case.addClassCleanup if isinstance(case, type) else case.addCleanup
        cleanup(scratch.cleanup)
        directory = Path(scratch.name)
    return directory / name
