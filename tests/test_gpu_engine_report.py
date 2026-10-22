"""Authored receipt-accounting fixtures; no device or scientific measurement."""
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_engine import report


class ReportTests(unittest.TestCase):
    def setUp(self):
        pool='authored-pool';weights={'checkpoint_sha256':'authored-champion'}
        self.docs={name:dict(schema=schema,source_commit=name+'-old-source',protocol_completed=True,
            team_pool_sha256=pool,device='authored GPU',jax='authored',weights=weights,rows=[])
            for name,schema in report.SCHEMAS.items()}
        # The original expansion schema predates the protocol_completed field;
        # its three explicit sizes/accounting are checked, not a fabricated flag.
        del self.docs['expansion']['protocol_completed']
        for b in report.BATCHES:
            self.docs['games']['rows'].append(dict(batch=b,status='COMPLETE_COHORT',completed_games=b,censored_games=0,
                completed_mask=[True]*b,final_turns=[50]*b,mean_completed_game_turns=50,
                elapsed_seconds=2,lockstep_requests=100,games_per_second=b/2,
                mean_end_to_end_request_ms=20,live_game_requests=b*50,live_game_requests_per_second=b*25,
                ordinary_game_turns=b*50,ordinary_game_turns_per_second=b*25,
                engine={'mean_ms':1},encode={'mean_ms':2},forward={'mean_ms':4}))
            self.docs['refill']['rows'].append(dict(batch=b,status='WINDOW_COMPLETE',games_per_second=10,
                measured=dict(completed_games=10,capped_games=1,elapsed_seconds=1)))
        for n in (256,1024,4096):
            self.docs['expansion']['rows'].append(dict(nodes=n,mean_ms=2,child_slots=81*n,
                valid_child_transitions=80*n,invalid_frozen_slots=n,node_expansions_per_second=n*500,
                valid_child_transitions_per_second=80*n*500))
        self.docs['cpu']={name:dict(games=2,incomplete_games=0,wall_s=seconds,games_per_second=2/seconds,
            decisions=10,outcomes=[dict(completed=True)]*2) for name,seconds in (('engine',4),('showdown',8))}
        f=self.docs['fidelity'];statuses=['MATCHED_THROUGH_TURN20']*41+['STATE_MISMATCH']*42+['LEGAL_ACTION_MISMATCH']
        statuses+=['UNRESOLVED_CHANCE_ROLE']*99+['UNRESOLVED_INITIAL_CHANCE']*17
        f.update(roster=dict(first_game=900,games=200),rows=[dict(game_id=900+i,team_ids=[1800+2*i,1801+2*i],status=s)
            for i,s in enumerate(statuses)],status_counts=dict(Counter(statuses)),matched=41,upper_nonmatch_fraction=.795)
        self.docs['ledger'].update(pool_sha256=pool,game_qualified_move_share=None,
            weighted_status_shares=dict(SELECTED_EFFECT_IMPLEMENTED=.9,PARTIAL=.05,PLACEHOLDER=.05))
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.paths={name:Path(self.temp.name)/(name+'.json') for name in self.docs};self.save()

    def save(self):
        for name,doc in self.docs.items():self.paths[name].write_text(json.dumps(doc))

    def test_tables_preserve_sources_counts_and_no_qualification(self):
        before={k:p.read_bytes() for k,p in self.paths.items()};out=report.assemble(self.paths)
        self.assertEqual(out['rows'][2]['games_per_second'],1024)
        self.assertEqual(out['rows'][2]['nominal_slot_requests_per_second'],102400)
        self.assertNotEqual(out['rows'][2]['nominal_slot_requests_per_second']/50,out['rows'][2]['games_per_second'])
        self.assertEqual(out['refill_rows'][2]['games_per_second'],10)
        self.assertEqual(out['sources']['games']['source_commit'],'games-old-source')
        self.assertIsNone(out['sources']['cpu']['source_commit'])
        self.assertEqual(out['recommendation'],'DO_NOT_ADVANCE_TO_PRODUCTION_NOW')
        self.assertIsNone(out['fidelity']['covered_subset_divergence_rate'])
        self.assertIsNone(out['coverage']['game_qualified_move_share'])
        self.assertEqual(out['fidelity']['fixed_denominator'],200)
        self.assertEqual(out['fidelity']['nonmatched_or_unresolved'],159)
        for k,p in self.paths.items():
            self.assertEqual(before[k],p.read_bytes())
            self.assertEqual(out['sources'][k]['sha256'],hashlib.sha256(before[k]).hexdigest())
        text=report.render_markdown(out)
        for b in report.BATCHES:self.assertIn('| '+str(b)+' |',text)
        for phrase in ('not additive','not a new','unavailable','CPU receipt does not embed','Mac/Metal'):
            self.assertIn(phrase,text)

    def test_pool_champion_hardware_and_partial_receipts_fail_closed(self):
        original=copy.deepcopy(self.docs)
        for name,key,value in (('refill','team_pool_sha256','other'),('refill','weights',{}),
            ('expansion','device','other'),('games','protocol_completed',False),('fidelity','schema','wrong')):
            self.docs=copy.deepcopy(original);self.docs[name][key]=value;self.save()
            with self.assertRaises(ValueError):report.assemble(self.paths)

    def test_counts_rates_censoring_and_fixed_rosters_reconcile(self):
        original=copy.deepcopy(self.docs)
        cases=[('games',lambda d:d['rows'][0].update(games_per_second=99)),
            ('games',lambda d:d['rows'][0].update(censored_games=1)),
            ('games',lambda d:d['rows'][0]['completed_mask'].__setitem__(0,False)),
            ('fidelity',lambda d:d['rows'][0].update(game_id=999)),
            ('fidelity',lambda d:d['rows'][0].update(team_ids=[0,1])),
            ('fidelity',lambda d:d.update(matched=42)),
            ('expansion',lambda d:d['rows'][0].update(invalid_frozen_slots=0)),
            ('ledger',lambda d:d.update(weighted_status_shares=dict(SELECTED_EFFECT_IMPLEMENTED=1.2,PARTIAL=-.2,PLACEHOLDER=0)))]
        for name,mutate in cases:
            self.docs=copy.deepcopy(original);mutate(self.docs[name]);self.save()
            with self.assertRaises(ValueError):report.assemble(self.paths)

    def test_nonfinite_json_and_missing_inputs_are_not_reports(self):
        self.paths['cpu'].write_text('{"bad": NaN}')
        with self.assertRaises(ValueError):report.assemble(self.paths)
        with self.assertRaises(ValueError):report.assemble({'games':self.paths['games']})

    @unittest.skipUnless(importlib.util.find_spec('jax'),'CLI package also supports GPU timing')
    def test_canonical_cli_report_mode_never_calls_a_device_and_is_exclusive(self):
        from gpu_engine import benchmark
        prefix=Path(self.temp.name)/'report'
        argv=['benchmark','--out-prefix',str(prefix)]
        for name,path in self.paths.items():argv+=['--report-'+name,str(path)]
        with patch('sys.argv',argv),patch.object(benchmark,'sole_gpu',side_effect=AssertionError('device touched')):
            benchmark.main()
            self.assertEqual(json.loads(prefix.with_suffix('.json').read_text())['status'],'HISTORICAL_REPORT_NO_NEW_MEASUREMENT')
            self.assertTrue(prefix.with_suffix('.md').exists())
            with self.assertRaises(FileExistsError):benchmark.main()


if __name__=='__main__':unittest.main()
