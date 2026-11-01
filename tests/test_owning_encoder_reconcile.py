"""Stored fake records only: no process, signal, model or game."""
from copy import deepcopy
from pathlib import Path
import unittest

from pokezero import owning_encoder_reconcile as r
from pokezero import owning_encoder_study as s
from tests.test_owning_encoder_continuation import ContinuationTests


def stat_literal(process):
    fields=['0']*50
    for index,value in ((0,process['state']),(1,process['ppid']),(2,process['pgid']),
            (3,process['sid']),(19,process['start_ticks']),(49,process['exit_code'])):
        fields[index]=str(value)
    return str(process['pid'])+' (unit python) '+' '.join(fields)+'\n'


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.f = ContinuationTests()
        self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.plan = self.f.plan()
        self.plan_ref = self.f.fixture.put('paused-plan.json',self.plan)
        ledger = deepcopy(self.f.ledger)
        ledger['cells'][325]['status'] = 'ATTEMPTED'  # An older unknown must remain byteexact.
        ledger['continuation'] = dict(schema=r.c.SCHEMA,plan_sha256=self.plan_ref['sha256'],dispatch_order=[326,327])
        for i,pair in ((326,[0,1]),(327,[2,3])):
            ledger['cells'][i].update(status='ATTEMPTED',attempt=dict(index=i,slot=i-326,
                cpu_pair=pair,started_epoch=999))
        self.ledger = ledger
        self.ledger_ref = self.f.fixture.put('paused-ledger.json',ledger)
        output = Path('/original/attempt')
        controller = dict(pid=41,ppid=1,pgid=1,sid=1,start_ticks=99,state='T',
            exit_code=19,args=['python','-m','pokezero.owning_encoder_continuation','run',
                '/original/plan.json',self.plan_ref['sha256'],str(output)])
        workers = [dict(pid=100+i,ppid=41,pgid=100+i,sid=100+i,start_ticks=1000+i,state='S',exit_code=0,
            args=['python','-m','pokezero.owning_encoder_continuation','worker',str(output/'plan.json'),
                str(i),str(pair[0]),str(pair[1]),str(output/f'cell-{i}.json')]) for i,pair in ((326,[0,1]),(327,[2,3]))]
        self.pause = dict(kind='OWNED_CONTROLLER_ONLY_SIGSTOP_NO_WORKER_SIGNAL',
            observed_epoch=1000,ledger_sha256=self.ledger_ref['sha256'],dispatch_order=[326,327],
            controller_stopped=controller,controller_before={**controller,'state':'S','exit_code':0},
            workers_before=workers,current_attempted=[dict(index=i,attempt=ledger['cells'][i]['attempt']) for i in (326,327)])
        pause_ref = self.f.fixture.put('quiesce.json',self.pause)
        self.end = dict(schema=r.SCHEMA,controller_closed=True,controller_pid=41,controller_start_ticks=99,
            quiesce_sha256=pause_ref['sha256'],ledger_sha256=self.ledger_ref['sha256'],
            plan_sha256=self.plan_ref['sha256'],observed_epoch=1010,workers=[])
        for before in workers:
            i=int(before['args'][-4]);pair=ledger['cells'][i]['attempt']['cpu_pair']
            process={**before,'state':'Z','exit_code':0,'args':[]}
            process['stat_literal']=stat_literal(process)
            self.end['workers'].append(dict(index=i,process=process,
                remaining_group_processes=[],receipt=self.f.fixture.put(f'cell-{i}.json',self.f.receipt(i,pair)),
                attempted_marker=self.f.fixture.put(f'cell-{i}.attempted',dict(index=i,plan_sha256=self.plan_ref['sha256']))))
        self.refs = dict(plan=self.plan_ref,ledger=self.ledger_ref,quiesce=pause_ref)

    def invoke(self,end=None):
        return r.reconcile(**self.refs,terminal=self.f.fixture.put('terminal.json',self.end if end is None else end))

    def test_completed_receipts_retained_only_after_actual_zombie_zero_and_group_drain(self):
        result=self.invoke()
        self.assertEqual(result['reconciliation']['retained_completed_indices'],[326,327])
        self.assertEqual(result['recorded_games'],327)
        for i in range(2000):
            if i not in (326,327):self.assertEqual(result['cells'][i],self.ledger['cells'][i])
        self.assertEqual(result['cells'][326]['timing'],self.f.receipt(326,[0,1])['timing'])
        self.assertEqual(s.read_bound(self.ledger_ref),self.ledger)
        self.assertFalse(result['B3_qualified'])

    def test_unavailable_or_nonzero_child_status_cannot_become_completed_game(self):
        end=deepcopy(self.end);end['workers'][0]['process']['exit_code']=256
        end['workers'][0]['process']['stat_literal']=stat_literal(end['workers'][0]['process'])
        result=self.invoke(end)
        self.assertEqual(result['cells'][326]['status'],'UNKNOWN_ATTEMPTED')
        self.assertEqual(result['reconciliation']['retained_completed_indices'],[327])
        for change in (dict(state='S'),dict(start_ticks=123),dict(ppid=1),dict(args=['wrong'])):
            end=deepcopy(self.end);end['workers'][0]['process'].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.invoke(end)
        for field,value in (('remaining_group_processes',[{'pid':555}]),('index',328)):
            end=deepcopy(self.end);end['workers'][0][field]=value
            with self.subTest(field=field),self.assertRaises((ValueError,KeyError)):self.invoke(end)
        for field,value in (('stat_literal',None),('exit_code',256),('args',self.pause['workers_before'][0]['args'])):
            end=deepcopy(self.end);end['workers'][0]['process'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.invoke(end)

    def test_foreign_hash_marker_terminal_and_duplicate_never_recover(self):
        for field,value in (('quiesce_sha256','0'*64),('controller_closed',False),('controller_pid',42),('observed_epoch',999)):
            end=deepcopy(self.end);end[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.invoke(end)
        end=deepcopy(self.end);end['workers'].append(deepcopy(end['workers'][0]))
        with self.assertRaises(ValueError):self.invoke(end)
        marker=self.end['workers'][0]['attempted_marker'];Path(marker['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'artifact hash'):self.invoke()


if __name__=='__main__':unittest.main()
