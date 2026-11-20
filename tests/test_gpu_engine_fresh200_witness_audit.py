"""Cross-language closed-audit guards using bytes emitted by the Node core."""
import base64
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from gpu_engine import fresh200_witness_audit as audit


class WitnessAuditGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory();root = Path(cls.fixture.name)
        node = os.environ.get('POKEZERO_NODE', 'node')
        result = subprocess.run([node,str(Path(__file__).with_name('fresh200_witness_fixture.cjs')),str(root)],capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Node witness fixture failed: '+result.stderr)
        cls.manifest = json.loads((root/'manifest.json').read_text())
        cls.audit_bytes = (root/'audit.jsonl').read_bytes();cls.receipt_bytes = (root/'receipts.jsonl').read_bytes()

    @classmethod
    def tearDownClass(cls):cls.fixture.cleanup()

    def verify(self, data=None, receipts=None, manifest=None):
        manifest = manifest or self.manifest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);a=root/'audit';r=root/'receipts'
            a.write_bytes(self.audit_bytes if data is None else data);r.write_bytes(self.receipt_bytes if receipts is None else receipts)
            return audit.verify_files(a,r,manifest['expected_commands'],source_registry_hash=manifest['source_registry_hash'],
                call_operations=manifest['call_operations'])

    def rechain(self, mutate):
        """Keep a synthetic mutant's hashes coherent to exercise semantic gates."""
        rows=[json.loads(x) for x in self.audit_bytes.splitlines()];receipts=[json.loads(x) for x in self.receipt_bytes.splitlines()]
        manifest=copy.deepcopy(self.manifest);mutate(rows,receipts,manifest)
        blobs=[];parent=audit.ZERO;previous_receipt=audit.ZERO;offset=0
        for command,receipt in zip(manifest['expected_commands'],receipts):
            command['control']['previous_receipt_hash']=previous_receipt;receipt['control']=copy.deepcopy(command['control'])
            chunk=rows[offset:offset+receipt['rowCount']];encoded=[]
            for row in chunk:
                row['sequence']=len(blobs);row['previousHash']=parent
                wire=(json.dumps(row,separators=(',',':'),ensure_ascii=False)+'\n').encode();encoded.append(wire);blobs.append(wire);parent=audit.digest(wire)
            receipt['byteCount']=sum(map(len,encoded));receipt['chainHash']=parent
            wire=(json.dumps(receipt,separators=(',',':'),ensure_ascii=False)+'\n').encode();previous_receipt=audit.digest(wire);offset+=len(chunk)
        return b''.join(blobs),b''.join((json.dumps(r,separators=(',',':'),ensure_ascii=False)+'\n').encode() for r in receipts),manifest

    def test_exact_node_utf8_bytes_and_contiguous_closed_commands_pass_without_admission(self):
        result,commands=self.verify()
        self.assertEqual((result['command_count'],result['row_count']),(3,12))
        self.assertTrue(result['audit_rules_passed']);self.assertFalse(result['qualification']);self.assertEqual(len(commands),3)

    def test_missing_reordered_changed_or_extra_audit_bytes_fail(self):
        lines=self.audit_bytes.splitlines(keepends=True)
        for value in (b''.join(lines[1:]),b''.join([lines[1],lines[0],*lines[2:]]),self.audit_bytes.replace(b'tackle',b'TACKLE',1),self.audit_bytes+b'{}\n'):
            with self.subTest(size=len(value)),self.assertRaises(ValueError):self.verify(data=value)
        with self.assertRaises(ValueError):self.verify(receipts=self.receipt_bytes+self.receipt_bytes.splitlines(keepends=True)[0])

    def test_self_consistent_hashes_cannot_hide_bad_call_parent_gap_or_native_throw(self):
        def parent(rows,_receipts,_manifest):rows[1]['frame']['parentCall']=99
        def gap(rows,_receipts,_manifest):rows[1]['frame']['callId']=1
        def throwing(rows,_receipts,_manifest):rows[2]['operation']='CALL_THROW'
        def unknown(rows,_receipts,_manifest):rows[1]['frame']['operation']='unbound.callback'
        for mutate in (parent,gap,throwing,unknown):
            data,receipts,manifest=self.rechain(mutate)
            with self.subTest(mutation=mutate.__name__),self.assertRaises(ValueError):self.verify(data,receipts,manifest)

    def test_poisoned_open_false_source_or_qualifying_receipts_fail(self):
        for key,value in (('poisoned',True),('openFrameCount',1),('qualification',True),('sourceRegistryHash','4'*64),('hookIdentityDigest','5'*64)):
            receipts=[json.loads(x) for x in self.receipt_bytes.splitlines()];receipts[0][key]=value
            data=b''.join((json.dumps(x,separators=(',',':'))+'\n').encode() for x in receipts)
            with self.subTest(field=key),self.assertRaises(ValueError):self.verify(receipts=data)

    def test_exact_command_roster_and_literal_integer_types_are_required(self):
        for mutate in (lambda x:x['expected_commands'].pop(),lambda x:x.update(expected_commands=[]),lambda x:x['expected_commands'][0]['control'].update(stream=True),
                lambda x:x['expected_commands'][1]['control'].update(command=3),lambda x:x['expected_commands'][1]['control'].update(nonce=x['expected_commands'][0]['control']['nonce'])):
            manifest=copy.deepcopy(self.manifest);mutate(manifest)
            with self.assertRaises(ValueError):self.verify(manifest=manifest)

    def test_duplicate_keys_nonfinite_and_raw_negative_zero_are_not_literal_audit_evidence(self):
        for value in (b'{"a":1,"a":2}',b'{"x":NaN}',b'{"x":Infinity}',b'{"x":-0}',b'{"x":-0.0}'):
            with self.subTest(value=value),self.assertRaises(ValueError):audit.document(value)

    def test_snapshot_wire_keeps_node_float_encoding_and_rejects_changed_state_or_hash(self):
        packet=self.manifest['packet'];snapshot=self.manifest['snapshot']
        self.assertIn(b'1e-7',base64.b64decode(packet['state_wire_base64']))
        self.assertIs(audit.verify_snapshot_wire(packet,snapshot),packet)
        changed=copy.deepcopy(snapshot);changed['battle']['turn']+=1
        with self.assertRaises(ValueError):audit.verify_snapshot_wire(packet,changed)
        changed=dict(packet,state_sha256='0'*64)
        with self.assertRaises(ValueError):audit.verify_snapshot_wire(changed,snapshot)
        changed=dict(packet,requests_wire_base64='***')
        with self.assertRaises(ValueError):audit.verify_snapshot_wire(changed,snapshot)


if __name__ == '__main__':unittest.main()
