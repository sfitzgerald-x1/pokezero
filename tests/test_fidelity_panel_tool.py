"""Pure-data controls for panel binding and conservative attribution."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SPEC=importlib.util.spec_from_file_location('panel_tool',Path(__file__).resolve().parents[1]/'scripts/fidelity_panel.py')
P=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class PanelToolTests(unittest.TestCase):
    def test_cli_preserves_harness_failure_but_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            shard=dict(source_commit='head',pool_sha256='pool',format_id='format',chance_tape_schema='tape',
                       panel=dict(first_game=1,games=1))
            for status in ('HARNESS_ERROR','STATE_MISMATCH','UNRESOLVED_CHANCE_ROLE','MATCHED_THROUGH_TURN20'):
                rows=[dict(game_id=1,status=status,error='injected bridge failure' if status=='HARNESS_ERROR' else None)]
                source=folder/(status+'.json');out=folder/(status+'-merged.json')
                source.write_text(json.dumps(shard|dict(rows=rows)))
                result=subprocess.run([sys.executable,str(P.__file__),'merge',str(out),str(source)],capture_output=True,text=True)
                failed=status=='HARNESS_ERROR'
                self.assertEqual(result.returncode,1 if failed else 0,result.stderr)
                artifact=json.loads(out.read_text())
                self.assertEqual(artifact['protocol_completed'],not failed)
                self.assertEqual(artifact['rows'],rows)
                self.assertEqual(result.stdout.startswith('INCOMPLETE:'),failed)

    def test_failed_shard_cannot_become_a_completed_match(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);source=folder/'shard.json';out=folder/'merged.json'
            source.write_text(json.dumps(dict(source_commit='head',pool_sha256='pool',format_id='format',
                chance_tape_schema='tape',panel=dict(first_game=1,games=1),protocol_completed=False,
                rows=[dict(game_id=1,status='MATCHED_THROUGH_TURN20')])))
            result=subprocess.run([sys.executable,str(P.__file__),'merge',str(out),str(source)],capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertFalse(json.loads(out.read_text())['protocol_completed'])

    def test_run_cli_propagates_protocol_failure(self):
        args=['panel','run','--tree','unused','--team-pool','unused','--expected-pool-sha','unused',
              '--showdown-root','unused','--node','unused','--out','unused']
        with mock.patch.object(sys,'argv',args),mock.patch.object(P,'run',return_value={'protocol_completed':False}):
            with self.assertRaises(SystemExit) as error:P.main()
        self.assertEqual(error.exception.code,1)

    def test_attribution_rejects_incomplete_panels_before_loading_pool(self):
        with tempfile.TemporaryDirectory() as folder:
            panel=Path(folder)/'panel.json'
            panel.write_text(json.dumps(dict(games=1,rows=[dict(game_id=1,status='HARNESS_ERROR')])))
            with self.assertRaisesRegex(ValueError,'incomplete panel'):
                P.attribute(argparse.Namespace(panel=panel,pool='missing',out=Path(folder)/'out.json'))

    def test_unknown_random_calls_do_not_acquire_a_convention_proof(self):
        for role, detail in [('SAMPLE',{'move':'metronome','length':5}),('TIE_ORDER',['surprise','surprise']),
                             ('MISSING_ACTOR','sample'),('NEW_ROLE',None)]:
            row=dict(status='UNRESOLVED_CHANCE_ROLE',steps=[dict(chance_audit=[dict(kind='UNRESOLVED',role=role,detail=detail)])])
            self.assertEqual(P.classify(row,[[],[]],set())[1],'u')

    def test_old_pinch_debug_line_does_not_attribute_a_new_request(self):
        before=['|turn|2','|debug|Swarm boost','|upkeep','|turn|3']
        row=dict(first_divergence=[dict(field='0:0:hp',gpu=10,showdown=20)],
                 pre_boundary=dict(showdown=dict(log_tail=before)),native_log_tail=before+['|move|p1a: Mew|Tackle|p2a: Mew'])
        self.assertEqual(P.state_cause(row,[[],[]])[1],'u')
        row['native_log_tail'].append('|debug|Swarm boost')
        self.assertEqual(P.state_cause(row,[[],[]])[:2],('Swarm','a'))

    def test_shards_bind_pool_format_and_tape(self):
        base=dict(source_commit='head',pool_sha256='pool',format_id='format',chance_tape_schema='tape',
                  panel=dict(first_game=1,games=2),rows=[])
        for key in ('pool_sha256','format_id','chance_tape_schema'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as folder:
                folder=Path(folder);a=folder/'a.json';b=folder/'b.json'
                a.write_text(json.dumps(base));b.write_text(json.dumps(base|{key:'different'}))
                with self.assertRaisesRegex(ValueError,key):
                    P.merge(argparse.Namespace(shards=[a,b],out=folder/'out.json'))


if __name__=='__main__':unittest.main()
