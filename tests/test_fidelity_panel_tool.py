"""Pure-data controls for panel binding and conservative attribution."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('panel_tool',Path(__file__).resolve().parents[1]/'scripts/fidelity_panel.py')
P=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class PanelToolTests(unittest.TestCase):
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
