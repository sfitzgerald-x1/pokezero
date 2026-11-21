"""No-update control lifecycle for a separately served v5 actor."""

import hashlib
from pathlib import Path

from . import checkpoint_self_moves, v5_training
from .s0_transfer_roster import digest


def tensor_digest(runtime):
    import numpy as np

    entries = {}
    for name, value in sorted(runtime.model.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        entries[name] = [list(array.shape), array.dtype.str, hashlib.sha256(array.tobytes()).hexdigest()]
    return digest(entries)


def write_actor(directory, runtime):
    from gpu_engine import v5_model

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    checkpoint, archive = root / "checkpoint.pt", root / "actor.npz"
    saved = v5_training.save_checkpoint(checkpoint, runtime)
    provenance = saved["encoding_receipt"]
    proof = None if runtime.self_moves else checkpoint_self_moves.off_from_receipt(provenance)
    export = v5_model.export(checkpoint, archive, expected_checkpoint_sha256=saved["sha256"], enable_v5=True,
                             self_moves=runtime.self_moves, off_provenance=proof)
    checkpoint.chmod(0o444)
    archive.chmod(0o444)
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=saved["sha256"], archive=str(archive),
                archive_sha256=export["weights_archive_sha256"], model_sha256=digest(runtime.model_config.metadata()),
                encoding_receipt=provenance, self_moves=runtime.self_moves, tensor_sha256=tensor_digest(runtime))


def reload_actor(actor):
    proof = None if actor["self_moves"] else checkpoint_self_moves.off_from_receipt(actor["encoding_receipt"])
    restored = v5_training.load_checkpoint(
        actor["checkpoint"], expected_checkpoint_sha256=actor["checkpoint_sha256"], enable_v5=True,
        expected_self_moves=actor["self_moves"], off_provenance=proof,
    )
    if (digest(restored.model_config.metadata()) != actor["model_sha256"]
            or tensor_digest(restored) != actor["tensor_sha256"] or restored.steps or restored.rows
            or restored.optimizer.state):
        raise ValueError("no-update checkpoint has changed model, buffers or update state")
    return restored


def shadow(directory, baseline, cycles, *, fixture, baseline_runtime=None):
    """Run the no-update save/export/reload path without a receipt chain."""
    if cycles not in (2, 50, 100):
        raise ValueError("expected 2 scratch, 50 cadence-1 or 100 cadence-2 cycles")
    from tempfile import TemporaryDirectory

    root = Path(directory)
    root.parent.mkdir(parents=True, exist_ok=True)
    runtime = reload_actor(baseline) if baseline_runtime is None else baseline_runtime
    if runtime.steps or runtime.rows or runtime.optimizer.state:
        raise ValueError("control must begin with the untouched baseline")
    expected_fixture = fixture(baseline, runtime.model_config)
    # Reuse a small temporary footprint. Only the served final control remains.
    for cycle in range(1, cycles):
        with TemporaryDirectory(dir=root.parent) as scratch:
            current = write_actor(Path(scratch) / "control", runtime)
            runtime = reload_actor(current)
            if current["tensor_sha256"] != baseline["tensor_sha256"]:
                raise ValueError(f"control weights changed at cycle {cycle}")
    control = write_actor(root, runtime)
    restored = reload_actor(control)
    if (control["tensor_sha256"] != baseline["tensor_sha256"]
            or fixture(control, restored.model_config) != expected_fixture):
        raise ValueError("final control differs from untouched baseline")
    return control
