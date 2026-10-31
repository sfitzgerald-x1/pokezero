"""Configuration/cache plumbing only, no real collection/model/training."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pokezero import v5_run as run
import test_v5_run as fixture


class DiscountTests(unittest.TestCase):
    def test_validated_default_and_explicit_manifest(self):
        original=fixture.plan()
        self.assertEqual(original.discount,1.)
        self.assertEqual(original.manifest()['discount'],1.)
        for value in (0,1,.9999):self.assertEqual(replace(original,discount=value).discount,float(value))
        for value in (-.1,1.1,float('nan'),float('inf'),True,'0.9999'):
            with self.subTest(value=value),self.assertRaises(ValueError):replace(original,discount=value)

    def test_original_missing_discount_declaration_resumes_only_at_one(self):
        plan=fixture.plan();legacy=plan.manifest();legacy.pop('discount')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);blob=run._json(legacy).encode();(root/'declaration.json').write_bytes(blob)
            event=dict(sequence=0,previous=None,time=0.,kind='RUN_STARTED',payload=dict(declaration_sha256=run._sha(legacy)))
            event['sha256']=run._sha(event);(root/'journal').mkdir();(root/'journal'/'000000000.json').write_text(run._json(event))
            restored=run.Store(root,plan,resume=True)
            self.assertEqual(len(restored.events),1);self.assertEqual((root/'declaration.json').read_bytes(),blob)
            with self.assertRaisesRegex(ValueError,'original immutable'):
                run.Store(root,replace(plan,discount=.9999),resume=True)

    def test_engine_cache_gets_exact_plan_discount_without_changing_targets(self):
        from gpu_engine import v5_selfplay as engine
        plan=replace(fixture.plan(),discount=.9999);batch=plan.iterations[0][0]
        collection=NS(decisions=('literal-row',),owning_histories=(),history_refusals=(),own_static_refusals=())
        backend=engine.EngineBackend(object(),object());prepared=dict(plan=plan,batch=batch,learner=object(),pairs=(),
            ledger=object(),indices=object(),streams=object(),gender=None)
        with tempfile.TemporaryDirectory() as tmp,patch.object(engine.base.driver,'construct',return_value=object()), \
                patch.object(engine,'collect_paired',return_value=collection),patch.object(engine.data,'write_cache') as cache, \
                patch.object(engine,'_file_sha',return_value='a'*64):
            result=backend.collect(prepared,NS(path=Path(tmp)),lambda *a,**k:None,lambda:None)
        self.assertEqual(result[1],'a'*64)
        self.assertEqual(cache.call_args.args[1],collection.decisions)
        self.assertEqual(cache.call_args.kwargs,dict(loss_config=plan.loss,batch_size=plan.train.batch_size,discount=.9999))


if __name__=='__main__':unittest.main()
