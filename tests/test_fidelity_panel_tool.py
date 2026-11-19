"""Pure-data controls for panel binding and conservative attribution."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

SPEC=importlib.util.spec_from_file_location('panel_tool',Path(__file__).resolve().parents[1]/'scripts/fidelity_panel.py')
P=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class PanelToolTests(unittest.TestCase):
    def test_panel_initializes_retained_genders_before_entry_and_supports_legacy_parent(self):
        for retained in (False, True):
            with self.subTest(retained=retained), tempfile.TemporaryDirectory() as folder:
                pool = Path(folder) / 'pool.npz';pool.write_bytes(b'pool')
                metadata = {'vocabulary': {}}
                archive = mock.MagicMock()
                archive.files = ['metadata_utf8']
                archive.__getitem__.return_value.tobytes.return_value = json.dumps(metadata).encode()
                np = mock.Mock()
                np.load.return_value.__enter__ = mock.Mock(return_value=archive)
                np.load.return_value.__exit__ = mock.Mock(return_value=False)
                jax, j, transition = mock.Mock(), mock.Mock(), mock.Mock()
                initial, gendered, table = mock.Mock(), mock.Mock(), object()
                transition.tables_from_metadata.return_value = table
                transition.state_from_pool.return_value._replace.return_value = initial
                events = []
                F = SimpleNamespace(reference_actions=SimpleNamespace())
                if retained:
                    def genders(state, data, showdown, node, roots):
                        self.assertIs(state, initial);self.assertEqual(data, metadata)
                        self.assertEqual((showdown, node), (Path('oracle'), 'node'))
                        events.append('genders');return gendered
                    F.initialize_comparison_genders = mock.Mock(side_effect=genders)
                def entry(state, actual_table):
                    self.assertIs(state, gendered if retained else initial)
                    self.assertIs(actual_table, table);events.append('entry')
                    raise RuntimeError('stop after entry boundary')
                transition.initialize.side_effect = entry
                args = argparse.Namespace(tree=folder, team_pool=pool, expected_pool_sha=hashlib.sha256(b'pool').hexdigest(),
                                          shard='1/2', first_game=50200, games=200, showdown_root='oracle', node='node')
                with mock.patch.object(P, '_engine', return_value=(jax,j,np,F,transition)), \
                     mock.patch.object(P, 'roster', return_value=([],{})), \
                     mock.patch.object(P.subprocess, 'check_output', side_effect=['', 'head']):
                    with self.assertRaisesRegex(RuntimeError, 'stop after entry boundary'):P.run(args)
                self.assertEqual(events, ['genders','entry'] if retained else ['entry'])
                if retained:self.assertEqual(np.arange.call_args, mock.call(50300,50400))

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

    def test_stale_form_events_do_not_override_current_damage(self):
        teams=[[dict(species='ditto',item='metalpowder'),dict(species='unown',item='twistedspoon'),
                dict(species='castform',item='leftovers')],[dict(species='mew',item='')]]
        for event in ('|-transform|p1a: Ditto|p2a: Mew',
                      '|-formechange|p1a: Castform|Castform-Rainy|[from] ability: Forecast'):
            before=['|turn|2',event,'|switch|p1a: Unown|Unown, L100|100/100','|turn|3']
            row=dict(first_divergence=[dict(field='1:0:hp',gpu=100,showdown=90)],
                     pre_boundary=dict(showdown=dict(log_tail=before)),post_boundary=dict(showdown=dict(active=[1,0])),
                     native_log_tail=before+['|move|p1a: Unown|Hidden Power|p2a: Mew','|-damage|p2a: Mew|90/100'])
            self.assertEqual(P.state_cause(row,teams)[:2],('Twisted Spoon (type-boost item)','a'))
            row['native_log_tail']=before+['|move|p2a: Mew|Tackle|p1a: Unown']
            self.assertEqual(P.state_cause(row,teams)[1],'u')

    def test_current_form_event_requires_affected_holder(self):
        for species,event,cause in (
            ('ditto','|-transform|p1a: Ditto|p2a: Mew','Transform'),
            ('castform','|-formechange|p1a: Castform|Castform-Rainy|[from] ability: Forecast','Forecast (Castform type change)')):
            teams=[[dict(species=species,item='')],[dict(species='mew',item='')]]
            before=['|turn|2']
            row=dict(first_divergence=[dict(field='0:0:hp',gpu=100,showdown=90)],
                     pre_boundary=dict(showdown=dict(log_tail=before)),post_boundary=dict(showdown=dict(active=[0,0])),
                     native_log_tail=before+[event,'|move|p2a: Mew|Tackle|p1a: '+species.title()])
            self.assertEqual(P.state_cause(row,teams)[:2],(cause,'a'))
            row['first_divergence'][0]['field']='1:0:hp'
            self.assertEqual(P.state_cause(row,teams)[1],'u')
            row['first_divergence'][0]['field']='0:0:hp'
            row['post_boundary']['showdown']['active']=[1,0]
            self.assertEqual(P.state_cause(row,teams)[1],'u')

    def test_truncated_request_log_requires_manual_attribution(self):
        row=dict(first_divergence=[dict(field='0:0:hp',gpu=100,showdown=90)],
                 pre_boundary=dict(showdown=dict(log_tail=['|turn|2'])),
                 native_log_tail=['|debug|Swarm boost','|-damage|p1a: Mew|90/100'])
        self.assertEqual(P.state_cause(row,[[],[]])[1],'u')

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
