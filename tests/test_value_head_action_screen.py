"""Synthetic numerical-producer controls. No model, saved corpus, or API access."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import tarfile
import unittest

P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('action_screen',P/'scripts/screen_value_head_actions.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class ScreenTests(unittest.TestCase):
    def candidates(self):
        return [dict(root_id=str(i),battle_id='battle-'+str(i//3),cost_seat=seat,split='train') for i in range(12) for seat in ('p1','p2')]

    def test_hash_rank_is_order_invariant_unique_balanced_and_excludes_old_roots(self):
        candidates=self.candidates();a=s.select_roots(candidates,{'1','2'},count=8)
        b=s.select_roots(list(reversed(candidates)),{'1','2'},count=8)
        self.assertEqual(a,b);self.assertEqual(len({r['root_id'] for r in a}),8)
        self.assertEqual([r['cost_seat'] for r in a],['p1','p2']*4)
        self.assertFalse({'1','2'} & {r['root_id'] for r in a})

    def test_outcome_action_value_fields_cannot_select_roots(self):
        candidates=self.candidates();a=s.select_roots(candidates,set(),count=8)
        for i,row in enumerate(candidates):row.update(outcome=i%2,value=100000-i,action=i%9,refusal=True)
        b=s.select_roots(candidates,set(),count=8)
        self.assertEqual([(r['root_id'],r['cost_seat'],r['search_seed']) for r in a],[(r['root_id'],r['cost_seat'],r['search_seed']) for r in b])

    def test_missing_or_duplicate_roots_refuse_without_refill(self):
        for candidates in (self.candidates()+[self.candidates()[0]],self.candidates()[:2]):
            with self.assertRaises(ValueError):s.select_roots(candidates,set(),count=8)
        with self.assertRaises(ValueError):s.select_roots(self.candidates(),{str(i) for i in range(12)},count=8)

    def test_full_denominators_and_unresolved_bounds(self):
        rows=[]
        for i in range(4):
            rows.append(dict(ordinal=i,paired_clean_search=i<2,chosen_actions_differ=i==0 if i<2 else None,
                heads={'champion':{'status':'SEARCHED','action':0},'refit':{'status':'SEARCHED' if i<2 else 'REFUSED','action':1 if i==0 else 0 if i<2 else None}}))
        report=s.summarize(rows,count=4)
        self.assertEqual((report['chosen_actions_differ'],report['paired_clean_searches'],report['unresolved_roots']),(1,2,2))
        self.assertEqual((report['full_roster_difference_fraction_lower'],report['full_roster_difference_fraction_upper']),(.25,.75))
        self.assertEqual(report['head_refusals'],{'champion':0,'refit':2})
        with self.assertRaises(ValueError):s.summarize(rows[:3],count=4)

    def test_native_witness_delegates_exact_arguments_once_and_preserves_reply(self):
        class Fold:
            def to_payload(self):return {'exact':'fold'}
        calls=[]
        class Native:
            def search_batched_multi_encoded(self,*args,**kwargs):
                calls.append((args,kwargs));return '{"side_one":[],"side_two":[],"prior_fallbacks":0,"capture_refusals":[]}'
        with tempfile.TemporaryDirectory() as d:
            witness=s.NativeWitness(Native(),d,s.canonical);fold=Fold();args=('state',2048,64,'tables',{},'ctx',fold,8,1.4,123,True,True)
            reply=witness.search_batched_multi_encoded(*args,test_kw=True)
            self.assertEqual(len(calls),1);self.assertIs(calls[0][0][6],fold);self.assertEqual(calls[0][0],args)
            self.assertEqual(calls[0][1],{'test_kw':True});self.assertTrue(witness.rows[0]['completed'])
            self.assertEqual(witness.rows[0]['native_seed'],123);self.assertEqual(json.loads(reply)['prior_fallbacks'],0)
            self.assertTrue((Path(d)/'native-00-input.json').is_file())

    def test_native_failure_retains_started_input_and_does_not_repeat(self):
        class Fold:
            def to_payload(self):return {}
        calls=[]
        class Native:
            def search_batched_multi_encoded(self,*args,**kwargs):calls.append(1);raise ValueError('authored refusal')
        with tempfile.TemporaryDirectory() as d:
            witness=s.NativeWitness(Native(),d,s.canonical)
            with self.assertRaisesRegex(ValueError,'authored refusal'):witness.search_batched_multi_encoded('x',1,1,'t',{},'c',Fold(),8,1.4,4)
            self.assertEqual(calls,[1]);self.assertFalse(witness.rows[0]['completed']);self.assertTrue((Path(d)/'native-00-input.json').exists())

    def test_roles_shapes_and_unselected_malformed_observations_fail(self):
        obs=dict(categorical_ids=[[0]*41 for _ in range(23)],numeric_features=[[0.]*132 for _ in range(23)],token_type_ids=[0]*23,
                 attention_mask=[True]*23,legal_action_mask=[True]+[False]*8)
        row=dict(requests={p:{'side':{'id':p},'active':[{}]} for p in ('p1','p2')},requested_players=['p1','p2'],
                 observations={p:copy.deepcopy(obs) for p in ('p1','p2')},actions={'p1':0,'p2':0})
        self.assertEqual(s.roles(row),['p1','p2'])
        for mutate in (lambda r:r['observations']['p2']['numeric_features'][0].pop(),lambda r:r['requested_players'].append('p1'),lambda r:r['actions'].update(p2=8)):
            bad=copy.deepcopy(row);mutate(bad)
            with self.assertRaises(ValueError):s.roles(bad)

    def test_freeze_reads_only192_train_battles_and_persists_before_any_search(self):
        obs=dict(categorical_ids=[[0]*41 for _ in range(23)],numeric_features=[[0.]*132 for _ in range(23)],token_type_ids=[0]*23,
                 attention_mask=[True]*23,legal_action_mask=[True]+[False]*8)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);roster=[];receipts={};contract={'authored':'no weights'}
            for i in range(256):
                battle=dict(index=i,battle_id='synthetic-'+str(i),seed=i,split='train' if i<192 else 'test',split_key=str(i));roster.append(battle)
                if i>=192:continue  # Any holdout read would fail: no directory exists.
                rows=[];prefix=[]
                for turn in range(2):
                    start=len(prefix);prefix.append('|turn|'+str(turn+1));requests={p:{'side':{'id':p},'active':[{}]} for p in ('p1','p2')}
                    row=dict(battle_id=battle['battle_id'],seed=i,decision_round_index=turn,protocol_start=start,protocol_end=len(prefix),protocol_delta=prefix[-1:],
                        public_prefix_sha256=s.digest(prefix),requests=requests,requested_players=['p1','p2'],observations={'p1':obs,'p2':obs},actions={'p1':0,'p2':0})
                    row['root_id']=s.digest([battle['battle_id'],turn,row['public_prefix_sha256'],requests]);rows.append(row)
                rows.append(dict(event='end',protocol_start=len(prefix),protocol_end=len(prefix),protocol_delta=[]))
                path=root/f'battle-{i:04d}';path.mkdir();blob=gzip.compress(b'\n'.join(s.canonical(r) for r in rows))
                (path/'boundaries.jsonl.gz').write_bytes(blob)
                receipt=dict(battle,contract=contract,files={'boundaries.jsonl.gz':s.file_sha(path/'boundaries.jsonl.gz')})
                s.write(path/'battle.json',receipt);receipts[f'battle-{i:04d}/battle.json']=s.file_sha(path/'battle.json')
            cohort=dict(battles=roster,config={'champion_sha256':'fake-sha'})
            s.write(root/'cohort.json',cohort)
            manifest=dict(status='COMPLETE_FIXED_COHORT',config=cohort['config'],cohort_sha256=s.file_sha(root/'cohort.json'),battle_receipt_hashes=receipts)
            s.write(root/'collection.json',manifest)
            plan=dict(collection=str(root),collection_sha256=s.file_sha(root/'collection.json'),cohort_sha256=s.file_sha(root/'cohort.json'),checkpoint_sha256='fake-sha',checkpoint_contract=contract,excluded_root_ids=[],source_bundle=str(root/'bank.tar.gz'))
            bank=s.freeze(plan,root/'bank.json')
            self.assertEqual(bank['count'],256);self.assertEqual(len({r['root_id'] for r in bank['roots']}),256)
            self.assertEqual(bank['search_calls'],0);self.assertTrue((root/'bank.json').is_file())
            self.assertFalse(any(int(name.split('/')[0].split('-')[-1])>=192 for name in bank['source_members'] if name.startswith('battle-')))
            with tarfile.open(root/'bank.tar.gz') as archive:
                self.assertEqual({m.name for m in archive.getmembers()}, {'root-bank.json'}|{'collection/'+n for n in bank['source_members']})
                self.assertTrue(all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in archive.getmembers()))
            with self.assertRaises(FileExistsError):s.freeze(plan,root/'bank.json')

if __name__=='__main__':unittest.main(verbosity=2)
