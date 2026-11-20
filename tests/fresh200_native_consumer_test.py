"""New saved-IO consumer guards; no prior guard suite or real game is rerun."""
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from gpu_engine.fresh200_native_consumer import ClosedCommandVerifier, CommandSpec, READERS


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf8')+b'\n'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def sha(value):return hashlib.sha256(value).hexdigest()


def raw(value):
    return dict(utf8_base64=base64.b64encode(value).decode('ascii'), bytes=len(value), sha256=sha(value),
        split_segments=len(value.decode('utf8').split('\n')))


def reseal(rows, receipt):
    """Adversarially recompute public hashes, without touching Native evidence."""
    parent=rows[0]['previous_hash'];first=rows[0]['sequence']
    for index, row in enumerate(rows):
        row['sequence']=first+index;row['previous_hash']=parent;row.pop('hash', None)
        row['hash']=sha(canonical(row));parent=row['hash']
    receipt['event_range']=[first, first+len(rows)];receipt['event_head']=parent
    receipt.pop('hash', None);receipt['hash']=sha(canonical(receipt))
    return b''.join(wire(x) for x in rows), wire(receipt)


class SavedConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The entry pins this resident executable before tests start. The
        # synthetic fixture is produced once, then reused for all mutations.
        cls.directory=tempfile.TemporaryDirectory(prefix='saved-consumer-')
        node=os.environ['FRESH200_TEST_NODE']
        result=subprocess.run([node, 'tests/fresh200_consumer_fixture.cjs', cls.directory.name],
            capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError('synthetic CJS fixture failed exit'+str(result.returncode)+': '+result.stderr.decode('utf8'))
        cls.root=Path(cls.directory.name);cls.manifest=json.loads((cls.root/'manifest.json').read_text())
        assert cls.manifest['synthetic_only'] and cls.manifest['no_actual_native_game']
        cls.blobs={name:(cls.root/(name+'.jsonl')).read_bytes() for name in ('audit','events','stdout')}
        cls.commands=[]
        for declaration in cls.manifest['commands']:
            item={}
            for key, value in declaration.items():
                data=(cls.root/value['path']).read_bytes()
                assert sha(data)==value['sha256'] and len(data)==value['bytes'];item[key]=data
            cls.commands.append(item)

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def verifier(self, **overrides):
        m=self.manifest
        values=dict(session=m['session'], battle_id=m['battle_id'], source_registry_hash=m['source_registry_hash'],
            expected_methods=m['expected_methods'], call_operations=set(m['call_operations']))
        values.update(overrides);return ClosedCommandVerifier(**values)

    def streams(self):return {name:io.BytesIO(value) for name, value in self.blobs.items()}

    def spec(self, index, **overrides):
        c=self.commands[index]
        values=dict(stdin_wire=c['stdin'], native_control_wire=c['native_control'], transport_control_wire=c['transport_control'],
            battle_id=self.manifest['battle_id'], hook_identity_digest=self.manifest['hook_identity_digest'])
        values.update(overrides);return CommandSpec(**values)

    def verify(self, verifier, streams, index, **overrides):
        c=self.commands[index];values=dict(spec=self.spec(index), native_receipt_wire=c['native_receipt'],
            transport_receipt_wire=c['transport_receipt'], **streams)
        values.update(overrides);return verifier.verify(**values)

    def command_events(self, index):
        receipt=json.loads(self.commands[index]['transport_receipt']);start,end=receipt['event_range']
        rows=[json.loads(x) for x in self.blobs['events'].splitlines()][start:end]
        return rows, receipt

    def verify_first_mutation(self, mutate, message=None):
        rows, receipt=self.command_events(0);mutate(rows,receipt)
        event_wire, receipt_wire=reseal(rows,receipt);streams=self.streams();streams['events']=io.BytesIO(event_wire)
        verifier=self.verifier()
        with self.assertRaisesRegex(ValueError, message or '.'):
            self.verify(verifier,streams,0,transport_receipt_wire=receipt_wire)
        self.assertEqual(verifier.completed,[]);self.assertEqual(verifier.native_cursor.sequence,0)
        self.assertEqual(verifier.transport_cursor.sequence,0);self.assertTrue(all(s.tell()==0 for s in streams.values()))

    def test_cjs_five_command_artifact_set_and_explicit_close(self):
        verifier=self.verifier();streams=self.streams()
        results=[self.verify(verifier,streams,i) for i in range(5)]
        result=verifier.finish_artifacts(**streams, expected_command_ids=[x['command_id'] for x in results])
        self.assertEqual(result['status'],'RAW_CLOSED_ARTIFACT_SET_VERIFIED_ONLY');self.assertFalse(result['qualification'])
        self.assertEqual(result['commands'],5)
        self.assertTrue(all(x['status']=='RAW_CLOSED_COMMAND_VERIFIED_ONLY' and not x['qualification'] for x in results))

    def test_literal_null_wait_duplicate_update_and_empty_chunk_generations(self):
        verifier=self.verifier();streams=self.streams();self.verify(verifier,streams,0)
        p1=verifier.transport_cursor.streams['p1'];p2=verifier.transport_cursor.streams['p2']
        self.assertEqual(p1.generation,3);self.assertEqual(p2.generation,1);self.assertTrue(p1.actionable);self.assertFalse(p2.actionable)
        self.assertEqual(p1.chunks[0]['value'],p1.chunks[1]['value']);self.assertEqual(p1.chunks[2]['value'],'')
        self.verify(verifier,streams,1);self.assertEqual(verifier.transport_cursor.streams['p1'].generation,4)
        self.assertFalse(verifier.transport_cursor.streams['p1'].actionable)

    def test_final_eof_summary_cannot_replace_missing_original_reader_eof(self):
        verifier=self.verifier();streams=self.streams()
        for index in range(3):self.verify(verifier,streams,index)
        rows,receipt=self.command_events(3);rows=[x for x in rows if not (x['operation']=='READER_EOF' and x['evidence']['stream']=='p2')]
        event_wire,receipt_wire=reseal(rows,receipt);streams['events']=io.BytesIO(event_wire)
        with self.assertRaisesRegex(ValueError,'EOF proof is incomplete'):
            self.verify(verifier,streams,3,transport_receipt_wire=receipt_wire)
        self.assertEqual(len(verifier.completed),3);self.assertFalse(verifier.transport_cursor.source_eof)

    def test_snapshot_state_must_equal_native_serializer_even_after_public_rehash(self):
        verifier=self.verifier();streams=self.streams()
        for index in range(2):self.verify(verifier,streams,index)
        rows,receipt=self.command_events(2)
        row=next(x for x in rows if x['operation']=='STDOUT_WRITE')
        packet=json.loads(base64.b64decode(row['evidence']['wire']['utf8_base64']));packet['snapshot']['battle']['turn']=2
        bad=wire(packet);row['evidence']['wire']=raw(bad)
        event_wire,receipt_wire=reseal(rows,receipt);streams['events']=io.BytesIO(event_wire);streams['stdout']=io.BytesIO(bad)
        with self.assertRaisesRegex(ValueError,'literal Native snapshot differs'):
            self.verify(verifier,streams,2,transport_receipt_wire=receipt_wire)
        self.assertEqual(len(verifier.completed),2)

    def test_independent_source_and_hook_admission_are_mandatory(self):
        for override in ({'source_registry_hash':'e'*64},):
            with self.assertRaisesRegex(ValueError,'incorrectly bound witness receipt'):
                self.verify(self.verifier(**override),self.streams(),0)
        with self.assertRaisesRegex(ValueError,'incorrectly bound witness receipt'):
            self.verify(self.verifier(),self.streams(),0,spec=self.spec(0,hook_identity_digest='e'*64))
        wrong=dict(self.manifest['expected_methods']);wrong['source.next']='e'*64
        with self.assertRaisesRegex(ValueError,'source methods differ'):
            self.verify(self.verifier(expected_methods=wrong),self.streams(),0)
        with self.assertRaises(ValueError):self.verifier(expected_methods={})

    def test_rehashed_completion_counts_do_not_establish_actual_drain(self):
        self.verify_first_mutation(lambda rows,r:r['streams']['p1'].__setitem__('consumed',999),'summary differs')

    def test_closed_event_schema_and_integer_ordinals_are_enforced(self):
        def unknown(rows, _):rows[1]['operation']='OBSERVER_INVENTED'
        def extra(rows, _):rows[1]['evidence']['invented']=True
        def boolean(rows, _):next(x for x in rows if x['operation']=='CONTROL_SCHEDULED')['evidence']['id']=False
        for mutate in (unknown,extra,boolean):
            with self.subTest(mutate=mutate.__name__):self.verify_first_mutation(mutate)

    def test_transport_request_call_must_join_actual_causal_owner(self):
        def wrong_owner(rows, _):
            next(x for x in rows if x['operation']=='REQUEST_CALL_BEGIN')['evidence']['side']='p2'
        self.verify_first_mutation(wrong_owner,'request producer join differs')
        def missing_native(rows, _):
            # All local transport references still match one another, but the
            # true Native call roster contains no call ID999.
            original=next(x for x in rows if x['operation']=='REQUEST_CALL_BEGIN')['evidence']['call_id']
            for row in rows:
                for key in ('call_id','request_call_id','emit_call_id'):
                    if type(row['evidence'].get(key)) is int and row['evidence'][key]==original:row['evidence'][key]=999
        self.verify_first_mutation(missing_native,'actual causal emitRequest rosters differ')

    def test_incomplete_event_file_rolls_back_both_cursors(self):
        verifier=self.verifier();streams=self.streams();streams['events']=io.BytesIO(self.blobs['events'][:100])
        with self.assertRaises(ValueError):self.verify(verifier,streams,0)
        self.assertEqual(verifier.native_cursor.sequence,0);self.assertEqual(verifier.transport_cursor.sequence,0)
        self.assertTrue(all(s.tell()==0 for s in streams.values()))

    def test_native_audit_corruption_preserves_saved_stream_offsets(self):
        verifier=self.verifier();streams=self.streams();streams['audit']=io.BytesIO(self.blobs['audit'].replace(b'COMMAND_BEGIN',b'COMMAND_BREAK',1))
        with self.assertRaisesRegex(ValueError,'begin missing'):self.verify(verifier,streams,0)
        self.assertTrue(all(s.tell()==0 for s in streams.values()));self.assertEqual(verifier.completed,[])

    def test_stdout_contamination_rollback_and_retry_reuses_same_artifacts(self):
        verifier=self.verifier();streams=self.streams();original=streams['stdout']
        streams['stdout']=io.BytesIO(b'{"type":"unexpected"}\n'+self.blobs['stdout'])
        with self.assertRaisesRegex(ValueError,'stdout wire differs'):self.verify(verifier,streams,0)
        self.assertEqual(verifier.completed,[]);self.assertTrue(all(s.tell()==0 for s in streams.values()))
        streams['stdout']=original;result=self.verify(verifier,streams,0)
        self.assertFalse(result['qualification']);self.assertEqual(len(verifier.completed),1)

    def test_independent_exact_stdin_and_completion_parent_cannot_be_adopted(self):
        for field in ('stdin_wire','transport_control_wire'):
            key='stdin' if field=='stdin_wire' else 'transport_control'
            value=self.commands[0][key]
            if field=='stdin_wire':bad=value[:-1]+b' \n'
            else:
                document=json.loads(value);document['previous_completion_hash']='f'*64;bad=wire(document)
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.verify(self.verifier(),self.streams(),0,spec=self.spec(0,**{field:bad}))

    def test_artifact_tail_roster_and_nonclose_prefix_are_not_final_closure(self):
        verifier=self.verifier();streams=self.streams();result=self.verify(verifier,streams,0)
        with self.assertRaisesRegex(ValueError,'explicit final close'):
            verifier.finish_artifacts(**streams,expected_command_ids=[result['command_id']])
        for index in range(1,5):self.verify(verifier,streams,index)
        ids=[x['command_id'] for x in verifier.completed]
        with self.assertRaisesRegex(ValueError,'complete command roster'):
            verifier.finish_artifacts(**streams,expected_command_ids=ids[:-1])
        for name in ('audit','events','stdout'):
            original=streams[name];streams[name]=io.BytesIO(b'UNACCOUNTED')
            with self.subTest(stream=name),self.assertRaisesRegex(ValueError,'unaccounted'):
                verifier.finish_artifacts(**streams,expected_command_ids=ids)
            self.assertEqual(streams[name].tell(),0);streams[name]=original


if __name__=='__main__':unittest.main()
