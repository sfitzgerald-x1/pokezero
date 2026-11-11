"""Inert declared-cell/claim/capture controls, no actual model or game."""
import asyncio
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pokezero import phase_d_ladder_panel as panel
from pokezero.live_protocol_actor import ActorContract


def declaration():
    rungs=[dict(name=n,worlds=w,chance_samples=c,depth=d,leaf=l)for n,w,c,d,l in (
        ('R1',2,1,1,'VALUE'),('R3',32,1,1,'VALUE'),('R4',32,8,1,'VALUE'),('R7',32,1,2,'VALUE'),('R5',8,1,1,'ROLLOUT'))]
    rungs[-1]['rollout_leaves']=8
    return dict(schema=panel.SCHEMA,checkpoint_sha256=ActorContract.champion().checkpoint_sha256,
        max_paired_cells=2000,screen_paired_cells=500,seeds=1000,seats=[0,1],foulplay_ms=1000,
        learner_limit_seconds=60,confidence=.9,screen_threshold_pp=3.,battle_seed_start=10000,
        foulplay_seed_start=20000,stream_start=30000,rungs=rungs,root_operator='ARGMAX_MEAN_COMPLETED_Q_COMMON_ALL_RUNGS')


class FixedRawPanel(unittest.TestCase):
    def test_same_pair_seeds_all_rungs_and_immutable_prefix(self):
        d=declaration();rows={arm:panel.cells(d,arm)for arm in panel.RUNG_NAMES}
        for arm,row in rows.items():
            self.assertEqual(row[:500],panel.cells(d,arm,limit=500))
            self.assertEqual([(x['seed'],x['fp_seed'],x['seat'])for x in row],[(x['seed'],x['fp_seed'],x['seat'])for x in rows['RAW']])
        self.assertEqual(len({x['ordinal']for row in rows.values()for x in row}),12000)
        bad=copy.deepcopy(rows['RAW'][0]);bad['seed']+=1
        with self.assertRaisesRegex(ValueError,'differs'):panel.checked_cell(d,bad)

    def test_wrong_budget_or_checkpoint_refuses(self):
        for key,value in [('foulplay_ms',500),('learner_limit_seconds',30),('checkpoint_sha256','a'*64),('screen_paired_cells',501)]:
            d=declaration();d[key]=value
            with self.assertRaises(ValueError):panel.validate(d)
        with self.assertRaises(ValueError):panel.cells(declaration(),'RAW',limit=501)

    def test_claim_precedes_runtime_and_no_replay_unknown_retained(self):
        with TemporaryDirectory()as directory:
            root=Path(directory);d=declaration();dp=root/'declaration.json';dp.write_bytes(panel.encoded(d));cell=panel.cells(d,'RAW')[0]
            request=dict(schema='pokezero.champion-search-ladder.raw-request.v1',run_directory=str(root),declaration=str(dp),
                declaration_sha256=panel.sha(d),cell=cell,checkpoint='/inert/champion.pt',checkpoint_sha256=d['checkpoint_sha256'],
                cpu_pair=[1,2],base_config={})
            path=root/'requests'/f"{cell['ordinal']}.json";panel.write_new(path,request)
            @contextmanager
            def capture(*a,**kw):yield
            class Runtime:
                async def run(self,config):
                    self.assert_started=(root/'starts'/f"{cell['ordinal']}.json").exists()
                    raise RuntimeError('inert interrupted original game')
            runtime=Runtime()
            with patch.object(panel,'configuration',return_value=NS(checkpoint=Path('/inert/champion.pt'))),\
                    patch.object(panel,'owning_capture',capture),patch('pokezero.live_protocol_runtime._checkpoint_sha',return_value=d['checkpoint_sha256']):
                self.assertEqual(asyncio.run(panel.raw_worker(path,panel.sha(request),runtime=runtime,expected_pair=(1,2))),0)
                self.assertTrue(runtime.assert_started)
                result=json.loads((root/'results'/f"{cell['ordinal']}.json").read_bytes())
                self.assertEqual(result['status'],'UNKNOWN');self.assertIsNone(result['score'])
                with self.assertRaises(FileExistsError):asyncio.run(panel.raw_worker(path,panel.sha(request),runtime=runtime,expected_pair=(1,2)))

    def test_request_binding_and_attempt_prefix(self):
        with TemporaryDirectory()as directory:
            p=Path(directory)/'x.json';panel.write_new(p,{'cell':1})
            self.assertEqual(panel.read_bound(p,panel.sha({'cell':1})),{'cell':1})
            with self.assertRaises(ValueError):panel.read_bound(p,'0'*64)
            with self.assertRaises(FileExistsError):panel.write_new(p,{'cell':2})


if __name__=='__main__':unittest.main()
