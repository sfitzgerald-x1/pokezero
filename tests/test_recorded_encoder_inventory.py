import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pokezero.mirror_value_probe import CHAMPION_SHA,canonical,cohort,digest
from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
from pokezero.recorded_encoder_inventory import inventory_boundaries,inventory_collection
from test_recorded_mirror_roots import boundary


class RecordedEncoderInventoryTests(unittest.TestCase):
    def rows(self,battle):
        row=boundary(battle);lines=['|turn|3','|-crit|p2a: Absol']+[
            '|request|'+json.dumps(row['requests'][seat]) for seat in ('p1','p2')]
        row.update(protocol_start=0,protocol_end=len(lines),protocol_delta=lines,public_prefix_sha256=digest(lines))
        row['root_id']=digest([battle['battle_id'],0,row['public_prefix_sha256'],row['requests']])
        end=dict(event='end',protocol_start=len(lines),protocol_end=len(lines)+1,protocol_delta=['|win|p1'])
        return [row,end]

    def collection(self,root):
        config=dict(root='inventory-authored-fixture',battles=8,champion_sha256=CHAMPION_SHA)
        roster=cohort(config['root'],config['battles']);hashes={}
        for battle in roster:
            dest=root/f"battle-{battle['index']:04d}";dest.mkdir()
            data=b''.join(canonical(row)+b'\n' for row in self.rows(battle))
            blob=gzip.compress(data);(dest/'boundaries.jsonl.gz').write_bytes(blob)
            # This is deliberately not a valid array and must remain unopened.
            (dest/'features.npz').write_text('not an array')
            receipt=dict(schema='pokezero.mirror_battle.v1',**battle,contract=CHAMPION_CONTRACT,
                files={'boundaries.jsonl.gz':hashlib.sha256(blob).hexdigest()})
            blob=canonical(receipt);(dest/'battle.json').write_bytes(blob)
            hashes[f"battle-{battle['index']:04d}/battle.json"]=hashlib.sha256(blob).hexdigest()
        c=canonical(dict(schema='pokezero.mirror_cohort.v1',battles=roster,roster_sha256=digest(roster),
            config=config,config_sha256=digest(config)));(root/'cohort.json').write_bytes(c)
        csha=hashlib.sha256(c).hexdigest()
        m=canonical(dict(schema='pokezero.mirror_collection.v1',status='COMPLETE_FIXED_COHORT',config=config,
            config_sha256=digest(config),cohort_sha256=csha,battles=roster,battle_receipt_hashes=hashes))
        (root/'collection.json').write_bytes(m)
        return dict(expected_collection_sha256=hashlib.sha256(m).hexdigest(),expected_cohort_sha256=csha)

    def test_full_fixed_inventory_keeps_every_kind_and_private_data_out_of_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);kwargs=self.collection(root);result=inventory_collection(root,**kwargs)
            self.assertEqual(result['battle_count'],8);self.assertEqual(result['split_counts'],{'test':1,'train':6,'validation':1})
            self.assertEqual(result['totals']['requested_views'],16)
            self.assertEqual(result['totals']['discarded_private_request_lines'],16)
            self.assertEqual(result['decision_nonrequest_kinds'],{'-crit':8,'turn':8})
            self.assertEqual(result['terminal_nonrequest_kinds'],{'win':8})
            self.assertFalse(result['public_wire_qualified']);self.assertFalse(result['encoder_parity_measured'])
            self.assertFalse(result['model_forward_measured']);self.assertFalse(result['feature_npz_or_models_opened']);self.assertTrue(result['json_reference_tensors_parsed'])
            self.assertNotIn('side',result)

    def test_hash_corruption_or_outside_symlink_fails_whole_collection(self):
        for mode in ('collection','cohort','receipt','boundary','symlink'):
            with tempfile.TemporaryDirectory() as directory,tempfile.TemporaryDirectory() as outside:
                root=Path(directory);kwargs=self.collection(root)
                path=root/({'collection':'collection.json','cohort':'cohort.json','receipt':'battle-0007/battle.json',
                    'boundary':'battle-0007/boundaries.jsonl.gz','symlink':'battle-0007/boundaries.jsonl.gz'}[mode])
                if mode=='symlink':
                    target=Path(outside)/'foreign.gz';target.write_bytes(path.read_bytes());path.unlink();path.symlink_to(target)
                else:path.write_bytes(path.read_bytes()+b' ')
                with self.assertRaises(ValueError):inventory_collection(root,**kwargs)

    def test_asymmetric_boundary_and_terminal_history_are_not_filtered(self):
        battle=dict(battle_id='asymmetric',seed=123);rows=self.rows(battle);first=rows[0]
        second=copy.deepcopy(first);second['decision_round_index']=1;second['requests']['p2']['wait']=True
        second['requested_players']=['p1'];second['paired']=False;second['observations'].pop('p2');second['actions'].pop('p2')
        delta=['|','|request|'+json.dumps(second['requests']['p2'])]
        second.update(protocol_start=first['protocol_end'],protocol_end=first['protocol_end']+2,protocol_delta=delta,
            public_prefix_sha256=digest(first['protocol_delta']+delta))
        second['root_id']=digest([battle['battle_id'],1,second['public_prefix_sha256'],second['requests']])
        end=dict(event='end',protocol_start=second['protocol_end'],protocol_end=second['protocol_end']+1,protocol_delta=['|win|p1'])
        report=inventory_boundaries(iter([first,second,end]),battle)
        self.assertEqual((report['boundaries'],report['requested_views'],report['asymmetric_boundaries']),(2,3,1))
        self.assertEqual(report['decision_nonrequest_kinds']['<blank>'],1)
        for bad in ([first,second],[first,end,second],[first,dict(end,protocol_start=0)],[None]):
            with self.assertRaises(ValueError):inventory_boundaries(iter(bad),battle)

    def test_attestation_covers_the_open_descriptor_even_if_path_is_restored(self):
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);kwargs=self.collection(root);target=(root/'battle-0007/boundaries.jsonl.gz').resolve()
            original=target.read_bytes();rows=[json.loads(line) for line in gzip.decompress(original).splitlines()]
            row=rows[0];row['protocol_delta'].insert(1,'|UNATTESTED-FIXTURE|');row['protocol_end']+=1
            row['public_prefix_sha256']=digest(row['protocol_delta'])
            row['root_id']=digest([row['battle_id'],0,row['public_prefix_sha256'],row['requests']])
            rows[-1]['protocol_start']+=1;rows[-1]['protocol_end']+=1
            alternate=target.with_name('alternate.gz');saved=target.with_name('authentic.gz')
            alternate.write_bytes(gzip.compress(b''.join(canonical(value)+b'\n' for value in rows)))
            real_open=Path.open;swapped=False
            def open_swapped(path,*args,**kw):
                nonlocal swapped
                if path==target and args and args[0]=='rb' and not swapped:
                    swapped=True;os.replace(target,saved);os.replace(alternate,target)
                    stream=real_open(path,*args,**kw)
                    os.replace(target,alternate);os.replace(saved,target)
                    return stream
                return real_open(path,*args,**kw)
            with patch.object(Path,'open',open_swapped):
                with self.assertRaisesRegex(ValueError,'streamed boundary artifact SHA'):
                    inventory_collection(root,**kwargs)
            self.assertTrue(swapped);self.assertEqual(target.read_bytes(),original)
