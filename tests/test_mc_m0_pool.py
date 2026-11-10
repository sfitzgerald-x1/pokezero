"""Handoff guards and full-roster collation, plus a Linux process-tree fixture."""
import copy
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import concurrent.futures
import threading
from gpu_engine import mc_m0 as m0, mc_m0_partition as part, mc_m0_pool as pool


def stage(index,root,k):
    y=[[1.]*k,[-1.]*k];status=[['terminal']*k]*2
    return dict(index=index,root=root,K=k,mode='unbatched' if k==64 else 'batched',outcomes=y,statuses=status,
                pairs={str(n):m0.pair_stats(y,status,n) for n in (64,256) if n<=k},cpu_seconds=1.,wall_seconds=1.)


class PoolTests(unittest.TestCase):
    def test_handoff_claim_is_global_across_different_pool_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);info=dict(pid=123,start='456')
            first=pool.claim_handoff(root,info,'cfg',root/'first')
            with self.assertRaises(FileExistsError):pool.claim_handoff(root,info,'cfg',root/'second')
            self.assertEqual(m0.sha(root/'handoff-claim.json'),first)

    def test_racing_controllers_cannot_both_claim_the_serial_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);barrier=threading.Barrier(2)
            def acquire(name):
                barrier.wait()
                try:return pool.claim_handoff(root,dict(pid=123,start='456'),'cfg',root/name)
                except FileExistsError:return None
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                results=list(executor.map(acquire,('first','second')))
            self.assertEqual(sum(value is not None for value in results),1)

    def test_killed_partition_records_terminal_cpu_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);serial=base/'serial';serial.mkdir();out=base/'failed-pool'
            config={};manifest=dict(config_sha256='cfg',source_sha256={})
            info=dict(pid=123,start='456');real_popen=subprocess.Popen
            def killed(command,**kwargs):
                return real_popen([sys.executable,'-c','import os,signal;sum(range(100000));os.kill(os.getpid(),signal.SIGKILL)'],**kwargs)
            def released(info,directory,reference,accounting,workers):
                accounting['capacity']={};return {},[],5.
            with patch.object(pool.part,'qualify',return_value=(config,{},manifest)), \
                    patch.object(pool.m0,'validate'),patch.object(pool,'require_serial',return_value=info), \
                    patch.object(pool,'release_serial',side_effect=released),patch.object(pool.subprocess,'Popen',side_effect=killed):
                with self.assertRaisesRegex(RuntimeError,'partition failed'):
                    pool.main(['--qualified-run',str(serial),'--serial-pid','123','--workers','1','--out',str(out)])
            summary=json.loads((out/'failure-summary.json').read_bytes())
            self.assertTrue(summary['terminal_accounting_complete'])
            self.assertGreater(summary['pool_process_tree_cpu_seconds'],0)
            self.assertEqual(summary['observed_serial_process_tree_cpu_seconds_before_handoff'],5.)
            self.assertEqual(summary['workers'][0]['exit_code'],-signal.SIGKILL)
            self.assertFalse((out/'readout.json').exists())

    def test_memory_guard_is_conservative_and_prevents_overcommit(self):
        with patch.object(pool.Path,'read_text',side_effect=['1000','400']):
            self.assertEqual(pool.capacity_guard([dict(rss_bytes=100)],4)['conservative_projected_bytes'],800)
        with patch.object(pool.Path,'read_text',side_effect=['1000','401']):
            with self.assertRaisesRegex(ValueError,'memory headroom'):pool.capacity_guard([dict(rss_bytes=100)],4)

    def test_dispatch_covers_exactly_unfinished_roots_without_overlap(self):
        manifest=dict(config_sha256='cfg',source_sha256={'core':'sha'})
        plan=dict(config_sha256='cfg',core_sha256={'core':'sha'},groups=[list(range(1,50))],
                  reuse={'0:64':{},'0:256':{}},serial_release=dict(status='CONTROLLED_HANDOFF_AFTER_COMPLETE_ROOT'))
        part.validate_dispatch(plan,tuple(range(1,50)),0,manifest)
        for bad in ('overlap','missing','completed','unreleased'):
            changed=copy.deepcopy(plan)
            if bad=='overlap':changed['groups'].append([1])
            if bad=='missing':changed['groups'][0].pop()
            if bad=='completed':changed['groups'][0].append(0)
            if bad=='unreleased':changed['serial_release']['status']='FAILED'
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                part.validate_dispatch(changed,tuple(changed['groups'][0]),0,manifest)

    def test_stage_checks_raw_outcomes_and_root_identity(self):
        root=dict(actions=[0,1],kind='self',seed=0);reference={0:dict(root=root)}
        row=stage(0,root,64);self.assertEqual(pool.validate_stage(row,reference),(0,64))
        row['outcomes'][0][0]=-1
        with self.assertRaisesRegex(ValueError,'diagnostics'):pool.validate_stage(row,reference)

    def fixture(self,base):
        serial=base/'serial';out=base/'pool';serial.mkdir();out.mkdir()
        roots=[dict(kind='self' if i<25 else 'foulplay',seed=i,actions=[0,1]) for i in range(50)]
        m0.write(serial/'config.json',dict(roots=roots))
        manifest=dict(config_sha256=m0.sha(serial/'config.json'),source_sha256={'core':'sha'})
        m0.write(serial/'manifest.json',manifest)
        reference={i:dict(index=i,status='READY',root=r) for i,r in enumerate(roots)}
        m0.write(serial/'prepared.json',list(reference.values()))
        m0.write(serial/'runtime-smoke.json',[dict(kind=k,status='PASS') for k in ('self','foulplay')])
        reuse={}
        for k in (64,256):
            path=serial/f'root-000-{k}.json';m0.write(path,stage(0,roots[0],k))
            reuse[f'0:{k}']=dict(path=str(path),sha256=m0.sha(path))
        indices=list(range(1,50));worker=out/'worker-0';worker.mkdir()
        for i in indices:
            for k in (64,256):m0.write(worker/f'root-{i:03d}-{k}.json',stage(i,roots[i],k))
        m0.write(worker/'partition-summary.json',dict(status='PARTITION_COMPLETE',assigned=indices,completed=indices,cpu_seconds=99.))
        m0.write(out/'worker-exits.json',[0])
        plan=dict(groups=[indices],reuse=reuse,serial_release=dict(observed_process_tree_cpu_seconds=5.),
                  config_sha256=manifest['config_sha256'],core_sha256={'core':'sha'},adapter_sha256='adapter')
        m0.write(out/'plan.json',plan)
        m0.write(worker/'assignment.json',dict(indices=indices,config_sha256=manifest['config_sha256'],core_sha256={'core':'sha'},
                 adapter_sha256='adapter',plan_sha256=m0.sha(out/'plan.json')))
        return serial,out,reference,plan

    def test_collect_requires_successful_exits_and_full_raw_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            serial,out,reference,plan=self.fixture(Path(tmp))
            pool.collect(out,serial,reference,plan,100.,20.)
            report=json.loads((out/'readout.json').read_bytes())
            self.assertTrue(report['experiment_completion_verified']);self.assertEqual(report['completed_both_modes'],50)
        with tempfile.TemporaryDirectory() as tmp:
            serial,out,reference,plan=self.fixture(Path(tmp))
            (out/'worker-exits.json').write_text('[1]')
            with self.assertRaisesRegex(ValueError,'worker exits'):pool.collect(out,serial,reference,plan,100.,20.)
            self.assertFalse((out/'collection').exists())
        with tempfile.TemporaryDirectory() as tmp:
            serial,out,reference,plan=self.fixture(Path(tmp))
            m0.write(out/'worker-0'/'failure-summary.json',{})
            with self.assertRaisesRegex(ValueError,'failure evidence'):pool.collect(out,serial,reference,plan,100.,20.)
            self.assertFalse((out/'collection'/'summary.json').exists())

    @unittest.skipUnless(platform.system()=='Linux','Linux /proc required')
    def test_controlled_release_only_signals_identified_process_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package=root/'gpu_engine';package.mkdir();(package/'__init__.py').write_text('')
            (package/'mc_m0.py').write_text('import subprocess,sys,time\np=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])\nprint(p.pid,flush=True)\ntime.sleep(60)\n')
            serial=root/'serial';serial.mkdir();r=dict(kind='self',seed=0,actions=[0,1]);reference={0:dict(root=r)}
            for k in (64,256):m0.write(serial/f'root-000-{k}.json',stage(0,r,k))
            env=dict(os.environ,PYTHONPATH=str(root))
            child=subprocess.Popen([sys.executable,'-u','-m','gpu_engine.mc_m0','--out',str(serial)],
                                   cwd=root,env=env,stdout=subprocess.PIPE,text=True)
            descendant=None
            try:
                descendant=int(child.stdout.readline())
                with self.assertRaisesRegex(ValueError,'output identity'):pool.require_serial(child.pid,root/'other')
                self.assertIsNone(child.poll())
                info=pool.require_serial(child.pid,serial)
                stages,incomplete,cpu=pool.release_serial(info,serial,reference)
                self.assertEqual(set(stages),{(0,64),(0,256)});self.assertEqual(incomplete,[])
                self.assertGreaterEqual(cpu,0);self.assertEqual(child.wait(timeout=5),-signal.SIGTERM)
                self.assertIsNone(pool.process(descendant))
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=5)
                if descendant is not None and pool.process(descendant) is not None:os.kill(descendant,signal.SIGTERM)
                if child.stdout is not None:child.stdout.close()
