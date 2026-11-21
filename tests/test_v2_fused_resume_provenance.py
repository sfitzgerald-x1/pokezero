"""Resume through the fused entry refuses mismatched run and checkpoint identity."""
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gpu_engine import v2_fused
from gpu_engine.v2_distributed import Distribution
from pokezero import checkpoint_self_moves, v2_trainer, v5_run
from test_v2_trainer import configurations

try:
    import torch
except ImportError:
    torch = None


def exercise_resume(root, distribution=None):
    model, loss, train = configurations()
    declaration = {'fixture': 'fused-resume', 'self_moves': False}
    plan = SimpleNamespace(model=model, loss=loss, train=train, self_moves=False, evaluation=None, iterations=(), games=(),
                           manifest=lambda: declaration)
    context = dict(plan=plan, declared={}, preparation_seconds=0)
    checkpoint = root / 'on.pt'
    snapshot = v5_run.Snapshot(-1, checkpoint.name, v2_fused.sha(checkpoint), 'initial.npz',
                              'a' * 64, v5_run._sha(model.metadata()))
    events = [{'kind': 'SNAPSHOT', 'payload': asdict(snapshot)}]
    argv = ['--config', 'unused.json', '--reference-run', str(root), '--torch-memory-fraction', '0']
    rank = 0 if distribution is None else distribution.rank
    if distribution is not None:
        args = v2_fused.argument_parser().parse_args([*argv, '--out', str(root)])
        distribution.bind(context, args)

    cases = {
        'self_moves': 'checkpoint self_moves differs from explicit expectation',
        'digest': 'original checkpoint SHA256 differs',
        'plan': 'original fused run declaration binding required',
        'placement': ('distributed fused run requires its original placement' if distribution is None
                      else 'resume requires identical source/runtime/rank/game/batch contract'),
    }
    assertions = unittest.TestCase()
    for case, message in cases.items():
        out = root / case
        state = dict(snapshots=[asdict(snapshot)], reports=[], row_order='file',
                     declaration_sha256=v5_run._sha(declaration), checkpoint_encoding_receipts={})
        if distribution is not None:
            state['distribution'] = distribution.contract
        if case == 'digest':
            state['snapshots'][0]['checkpoint_sha256'] = 'b' * 64
        elif case == 'plan':
            state['declaration_sha256'] = 'c' * 64
        elif case == 'placement':
            state['distribution'] = dict(state.get('distribution', {}), world_size=2)
        if rank == 0:
            out.mkdir()
            (out / checkpoint.name).hardlink_to(checkpoint)
            v2_fused.write_json(out / 'state.json', state)
        if distribution is not None:
            distribution.barrier()
        with (patch.object(v2_fused, 'prepare', return_value=context),
              patch.object(checkpoint_self_moves, 'verified_run', return_value=(None, None, events)),
              assertions.assertRaisesRegex(ValueError, message)):
            v2_fused.main([*argv, '--out', str(out)], distribution=distribution)


def exercise_rank(rank, root):
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(root / 'rendezvous').as_uri(), rank=rank,
                            world_size=4, timeout=timedelta(seconds=90))
    try:
        exercise_resume(root, Distribution(rank, 4, dist.group.WORLD))
    finally:
        dist.destroy_process_group()


@unittest.skipIf(torch is None, 'torch required')
class FusedResumeProvenanceTests(unittest.TestCase):
    def checkpoint(self, root):
        runtime = v2_trainer.create(*configurations(), enable_v5=True, device='cpu', self_moves=True)
        v2_trainer.save_checkpoint(root / 'on.pt', runtime)

    def test_serial_refuses_wrong_encoding_digest_plan_and_placement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.checkpoint(root)
            exercise_resume(root)

    def test_four_ranks_refuse_wrong_encoding_digest_plan_and_placement(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.checkpoint(root)
            mp.spawn(exercise_rank, args=(root,), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
