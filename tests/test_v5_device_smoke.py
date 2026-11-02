"""Finite real Plan/config source controls; no runtime/model/game execution."""
import unittest
from pathlib import Path
import tempfile
from contextlib import ExitStack
from unittest.mock import patch
from pokezero import v5_device_smoke as smoke
from tests import test_v5_phasec as fixtures


def bank():
    return dict(run_id='inert-smoke-20k',pool_sha256='a'*64,games=20000,batch_rows=2048,model_seed=1760000000,
        stream_start=1763000000,identity_seed_start=1762000000,constructor_seed_start=1764000000,sampling_seed_start=1765000000,
        gender_seed=1766000000,gender_root_start=1767000000,gender_domain=1768000000,selection_seed=1769000000,max_requests=4096)


class DeviceSmokeTests(unittest.TestCase):
    def test_exact20k_original_controls_disjoint_identity_no_production_substitute(self):
        plan=smoke.build_plan(bank(),fixtures.profile(),fixtures.vocabulary(),fixtures.category())
        self.assertEqual(len(plan.games),20000);self.assertEqual(len(plan.iterations),1)
        self.assertEqual([len(b.games)for b in plan.iterations[0]],[2048]*9+[1568])
        self.assertEqual(plan.train.seed,1760000000);self.assertEqual(plan.train.epochs,5)
        self.assertEqual(plan.model.arm,'B');self.assertEqual(plan.model.width,512);self.assertEqual(plan.model.layers,3)
        self.assertEqual(plan.discount,.9999);self.assertIsNone(plan.evaluation)
        self.assertEqual(plan.games[0].team_indices,(0,1));self.assertEqual(plan.games[-1].team_indices,(39998,39999))
        self.assertEqual(plan.games[-1].stream_id,1763019999)
        self.assertEqual(plan.iterations[0][-1].constructor_seed,1764000009)
        for key,value in [('games',19999),('model_seed',0),('batch_rows',4096)]:
            with self.assertRaises(ValueError):smoke.build_plan(dict(bank(),**{key:value}),fixtures.profile(),fixtures.vocabulary(),fixtures.category())

    def test_original_run_timing_does_not_change_training_cursor_or_exports(self):
        from tests import test_v5_run as fixture
        from pokezero import v5_run as runs
        from gpu_engine import v5_model
        with tempfile.TemporaryDirectory()as folder:
            trainer=fixture.FakeTraining();backend=fixture.FakeBackend();recipe=fixture.plan()
            def export(checkpoint,out,**kw):
                Path(out).write_bytes(b'inert-export:'+Path(checkpoint).read_bytes())
                return dict(weights_archive_sha256=runs._file_sha(out))
            with ExitStack()as stack:
                for name,callback in [('create',trainer.create),('save_checkpoint',trainer.save),('load_checkpoint',trainer.load),('train_caches',trainer.train)]:
                    stack.enter_context(patch.object(runs.training,name,side_effect=callback))
                stack.enter_context(patch.object(v5_model,'export',side_effect=export))
                result=runs.run(recipe,folder,backend=backend,enable_v5=True,measure_timings=True)
            self.assertEqual(result['status'],'COMPLETE_DECLARATION');self.assertEqual(result['steps'],6)
            store=runs.Store(folder,recipe,resume=True);phases=store.matching('RUNTIME_PHASE')
            self.assertEqual(sum(x['scope']=='TRAIN_CACHES_INCLUSIVE'for x in phases),6)
            self.assertEqual(sum(x['scope']=='ACTOR_EXPORT'for x in phases),3)
            self.assertTrue(all(x['status']=='COMPLETE'and x['wall_seconds']>=0 for x in phases))
            self.assertEqual([x['iteration']for x in store.matching('SNAPSHOT')],[-1,0,1])
            self.assertEqual(backend.collect_calls,[1,2,3,4])

    def test_disabled_smoke_never_reads_or_prepares_runtime(self):
        with self.assertRaisesRegex(ValueError,'explicit finite smoke'):smoke.execute({},'/tmp/inert-only')

if __name__=='__main__':unittest.main()
