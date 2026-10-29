import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from pokezero.mirror_value_probe import canonical,digest
from pokezero.recorded_encoder_collection import compare_collection
from pokezero.recorded_encoder_parity import SHAPES
import test_recorded_encoder_inventory as fixtures


class RecordedCollectionTests(unittest.TestCase):
    def prepare(self,root):
        fixture=fixtures.RecordedEncoderInventoryTests();kwargs=fixture.collection(root)
        manifest=json.loads((root/'collection.json').read_bytes());battles={r['battle_id']:r for r in manifest['battles']};calls=[]
        class AuthoredAdapter:
            def reset(self,*,stream_id,battle_id):
                calls.append(('reset',stream_id,battle_id));self.refs=fixture.rows(battles[battle_id])[0]['observations']
            def feed(self,lines):
                if any(line.startswith('|request|') for line in lines):raise AssertionError('private line leaked')
                calls.append(('feed',len(lines)))
            def encode(self,request,*,owner_side):
                seat=f'p{owner_side+1}'
                if request['side']['id']!=seat:raise AssertionError('foreign request')
                return {k:np.array(self.refs[seat][k],np.float32 if k=='numeric_features' else np.bool_ if k.endswith('mask') else np.int32) for k in SHAPES}
        return kwargs,AuthoredAdapter(),calls

    def test_complete_fixed_roster_and_no_forward_never_qualify_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);kwargs,adapter,calls=self.prepare(root)
            result=compare_collection(root,**kwargs,adapter=adapter,with_forward=False)
            self.assertEqual(len(result['battles']),8);self.assertEqual(result['metrics']['views'],16)
            self.assertEqual([x[1] for x in calls if x[0]=='reset'],list(range(1,9)))
            self.assertEqual(len([x for x in calls if x[0]=='feed']),8)
            self.assertFalse(result['metrics']['model_forward_complete'])
            self.assertFalse(result['full_encoder_admitted']);self.assertFalse(result['public_wire_qualified'])
            self.assertEqual(result['new_games'],0);self.assertEqual(result['simulator_steps'],0)

    def test_second_pass_attests_actual_descriptor_not_only_initial_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);kwargs,adapter,_=self.prepare(root);target=(root/'battle-0007/boundaries.jsonl.gz').resolve()
            original=target.read_bytes();rows=[json.loads(line) for line in gzip.decompress(original).splitlines()]
            row=rows[0];row['protocol_delta'].insert(1,'|UNATTESTED|');row['protocol_end']+=1
            row['public_prefix_sha256']=digest(row['protocol_delta'])
            row['root_id']=digest([row['battle_id'],0,row['public_prefix_sha256'],row['requests']])
            rows[-1]['protocol_start']+=1;rows[-1]['protocol_end']+=1
            alternate=target.with_name('alternate.gz');saved=target.with_name('authentic.gz')
            alternate.write_bytes(gzip.compress(b''.join(canonical(row)+b'\n' for row in rows)))
            real_open=Path.open;opens=0
            def switched(path,*args,**kw):
                nonlocal opens
                if path==target and args and args[0]=='rb':
                    opens+=1
                    if opens==2:
                        os.replace(target,saved);os.replace(alternate,target);stream=real_open(path,*args,**kw)
                        os.replace(target,alternate);os.replace(saved,target);return stream
                return real_open(path,*args,**kw)
            with patch.object(Path,'open',switched):
                with self.assertRaisesRegex(ValueError,'replayed boundary artifact SHA'):
                    compare_collection(root,**kwargs,adapter=adapter,with_forward=False)
            self.assertEqual(opens,2);self.assertEqual(target.read_bytes(),original)
