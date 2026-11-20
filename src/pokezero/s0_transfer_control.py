"""No-update control lifecycle for a separately served v5 actor."""

import hashlib
from pathlib import Path

from . import checkpoint_self_moves, v5_training
from .s0_transfer_roster import digest, write_once


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


def shadow(directory, baseline, iterations, *, dry_run, fixture):
    """Save, export, reload and verify one control state per completed iteration."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    if dry_run:
        if iterations != []:
            raise ValueError("scratch fixtures carry no training progress")
        cycles = [None, None]
    else:
        cycles = iterations
        if len(cycles) not in (50, 100):
            raise ValueError("control cadence requires 50 or 100 completed iteration receipts")
        before = 0
        for i, entry in enumerate(cycles, 1):
            if (entry["iteration"] != i or type(entry["completed_games"]) is not int
                    or entry["completed_games"] <= before):
                raise ValueError("completed iteration order/progress changed")
            before = entry["completed_games"]
    runtime = reload_actor(baseline)
    expected_fixture = fixture(baseline, runtime.model_config)
    receipts = []
    for i, iteration in enumerate(cycles, start=1):
        # No collection and no optimizer application in this path.
        current = write_actor(root / f"{i:03d}", runtime)
        runtime = reload_actor(current)
        actual_fixture = fixture(current, runtime.model_config)
        if current["tensor_sha256"] != baseline["tensor_sha256"] or actual_fixture != expected_fixture:
            raise ValueError("control artifact or fixture changed during the no-update lifecycle")
        receipt = dict(schema="pokezero.s0-control-cycle.r2", dry_run=dry_run, control=current,
                       baseline=baseline, fixture_sha256=actual_fixture, cycle=i,
                       training_games=0, completed_iteration=iteration)
        receipts.append(write_once(root / f"{i:03d}" / "cycle.json", receipt))
    return current, dict(dry_run=dry_run, training_games=0, shadow_cycles=len(cycles),
                         completed_iterations_sha256=digest(iterations), cycle_sha256=receipts,
                         last_fixture_sha256=expected_fixture)
