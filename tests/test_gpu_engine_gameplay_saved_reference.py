"""Tiny authored captures only; no Native, engine, model or sampler is run."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import finite_gameplay_comparison as comparison
from gpu_engine import finite_gameplay_comparison_saved as saved
from gpu_engine import engine_gameplay_trace as trace
from gpu_engine.public_inputs import PublicEvent, Kind, Phase
from test_gpu_engine_gameplay_stream_comparison import write_trace, result


def evidence(boundary):
    state = boundary['state']; sides = []
    for side in (0, 1):
        sides.append(dict(pokemon=[dict(set={'species':f'mon{side*6+slot+1}'},
            hp=state['hp'][0][side][slot], types=['Normal','Normal'], maxhp=100, status='', fainted=False,
            storedStats={key:100 for key in ('atk','def','spa','spd','spe')},
            boosts={key:0 for key in ('atk','def','spa','spd','spe','accuracy','evasion')},
            ability='pressure', item='', moveSlots=[dict(id='tackle', pp=10)]*4) for slot in range(6)]))
    return dict(battle=dict(sides=sides, field={'weather':''}, turn=state['turn'][0]),
        requested_players=[f'p{i+1}' for i,x in enumerate(boundary['requested']) if x],
        terminal=dict(winner='p1',capped=False) if state['done'][0] else None,
        original_snapshot_sha256='literal-synthetic-not-a-native-execution')


class Owner:
    fail_p2 = False
    calls = []
    def __init__(self, *args, owner, root_id, **kw): self.owner=owner; self.root=root_id; self.frame=0
    def append(self, lines, constructor=False):
        self.calls.append((self.root,self.owner,tuple(lines),constructor))
        if self.fail_p2 and self.owner == 'p2': raise ValueError('literal unit owner refusal')
        if not constructor: self.frame=int(lines[0].split('|')[-1])
        row=[0]*16; row[:2]=[1,self.frame+1]
        return dict(words=[row], refused_lines=[],raw_kinds={'frame':len(lines)})
    def observation(self):
        actual,_=result(frame=self.frame,hp=100-self.frame)
        return {name:getattr(actual.owners[0].encoded.observation,name) for name in comparison.original.SHAPES}
    def close(self):pass


def fixture(folder):
    path=folder/'trace'; measured=write_trace(path,batch=2)
    gpu={101:[],102:[]}
    for line in path.read_text().splitlines():
        row=json.loads(line)
        if row['kind'] in ('INITIAL','DECISION'):gpu[row['payload']['stream_id']].append(row['payload'])
    roots=[];rows=[];reducer=comparison.Reducer([101,102])
    for ordinal,stream in enumerate((101,102)):
        initial,*decisions=gpu[stream]; current=evidence(initial['boundary'])
        roots.append(dict(stream_id=stream,native_seed=700+ordinal,native_started=True,
            engine_status='COMPLETED',trace_decisions=2,initial_chance_audit=[],
            initial_state_status='AGREE',initial_state_differences=[]))
        rows.append(dict(kind='INITIAL',stream_id=stream,native_seed=700+ordinal,
            native_compared_state=current,native_owner_lines=[['|initial'],['|initial']]))
        for captured in decisions:
            frame=captured['request']+1
            old=dict(stream_id=stream,request=captured['request'],state_status='AGREE',
                engine_raw=captured['after']['raw'],native_before_state_sha256=comparison.original._sha(current),
                native_compared_state=evidence(captured['after']),before_state_differences=[],after_state_differences=[],
                prior_state_agreement=True,chance_audit=[],native_owner_lines=[[f'|frame|{frame}']]*2,
                owners=[comparison.owner_refusal([],TypeError('Object of type int32 is not JSON serializable')) for _ in (0,1)])
            current=old['native_compared_state']
            if stream==102 and frame==2:
                old.pop('native_compared_state');old['state_status']='UNKNOWN'
                old['error']=dict(type='ValueError',message='SPIKE_UNRESOLVED:TIE_ORDER')
                old['owners']=[dict(status='UNKNOWN',refusals=[dict(kind='NATIVE_ADVANCEMENT_REFUSAL',reason=old['error']['message'])])]*2
            reducer.add(old);rows.append(old)
    native_path=folder/'native-decisions';native_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    measurement_path=folder/'measurement';measurement_path.write_text(json.dumps(measured))
    native=dict(schema=comparison.SCHEMA,measurement_sha256=trace.digest_file(measurement_path),
        trace_sha256=trace.digest_file(path),runtime_binding=measured['runtime_binding'],pool_sha256='a'*64,
        native_seed_start=700,roots=roots,decisions_artifact={'sha256':trace.digest_file(native_path)},summary=reducer.summary())
    result_path=folder/'native-result';result_path.write_text(json.dumps(native))
    return dict(measurement_path=measurement_path,expected_measurement_sha256=trace.digest_file(measurement_path),
        trace_path=path,expected_trace_sha256=trace.digest_file(path),native_result_path=result_path,
        expected_native_result_sha256=trace.digest_file(result_path),native_decisions_path=native_path,
        expected_native_decisions_sha256=trace.digest_file(native_path),
        pool=NS(metadata=lambda:{'vocabulary':dict(abilities=['','pressure'],items=[''],moves=['','tackle'],types=['','normal','water'])}),
        catalog=object(),env_config=object(),out_path=folder/'reanalysed')


class SavedGameplayTests(unittest.TestCase):
    def run_saved(self, args):
        # Any accidental runtime path is a hard test failure, including packing.
        with patch.object(comparison.original.source_provenance,'_pool',return_value={'pool_sha256':'a'*64}), \
             patch.object(comparison,'OwnerReference',Owner), \
             patch.object(comparison.original,'OwningShowdownEnv',side_effect=AssertionError('Native forbidden')), \
             patch.object(comparison.original,'pool_sets',side_effect=AssertionError('sets forbidden')), \
             patch.object(comparison.original,'pack_sets',side_effect=AssertionError('packing forbidden')), \
             patch.object(comparison,'ChanceControl',side_effect=AssertionError('chance forbidden')):
            return saved.reanalyse(**args)

    def test_original_numpy_event_words_serialize_without_value_change(self):
        event=PublicEvent(Kind.TURN,np.int32(2**31-1),np.int32(0),Phase(0))
        words=event.words();self.assertEqual(words.dtype,np.int32)
        reference=comparison.OwnerReference.__new__(comparison.OwnerReference)
        reference.catalog=object();reference.owner='p1';reference.wire=comparison.public_wire.WireState()
        reference.poisoned=False;reference.history=[];reference.request_count=0;reference.last_request=None
        import hashlib
        reference.prefix=hashlib.sha256();reference.backend=None;reference.backend_error=None
        with patch.object(comparison.v4_encoder,'wire_kwargs',return_value={}), \
             patch.object(comparison.public_wire,'project_line',side_effect=lambda wire,line,**kw:(wire,NS(events=[event]))):
            packet=reference.append(['|turn|1','|request|'+json.dumps({'side':{'id':'p1'},'rqid':1})])
        self.assertEqual(packet['words'],[words.tolist()]);self.assertTrue(all(type(x) is int for x in packet['words'][0]))
        self.assertEqual(json.loads(json.dumps(packet))['words'],[words.tolist()])
        comparison.original._sha(packet['words'])

    def test_saved_full_roster_no_runtime_and_unresolved_call_unchanged(self):
        Owner.calls=[];Owner.fail_p2=False
        with tempfile.TemporaryDirectory() as temp:
            args=fixture(Path(temp)); original=args['native_decisions_path'].read_bytes()
            actual=self.run_saved(args);summary=actual['summary']
            self.assertEqual(summary['state_counts'],{'AGREE':3,'UNKNOWN':1})
            self.assertEqual(summary['original_owner_decision_views'],8)
            self.assertEqual(summary['conditional_on_continuous_measured_state_agreement']['matched'],6)
            self.assertEqual(summary['refusals_by_event_kind'],{'NATIVE_ADVANCEMENT_REFUSAL':2})
            self.assertEqual(actual['new_native_calls'],0);self.assertEqual(actual['new_chance_samples'],0)
            self.assertEqual(args['native_decisions_path'].read_bytes(),original)
            rows=[json.loads(line) for line in args['out_path'].read_text().splitlines()]
            self.assertEqual(rows[-1]['error']['message'],'SPIKE_UNRESOLVED:TIE_ORDER')
            self.assertEqual(len(Owner.calls),10) # 4 initial + 6 successful owner deltas
            for column in summary['numeric_columns']+summary['categorical_columns']:
                self.assertEqual(column['expected_cells'],184)
                self.assertEqual(column['compared_cells']+column['unknown_cells'],184)

    def test_owner_failure_remains_local_in_saved_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            args=fixture(Path(temp));Owner.fail_p2=True
            try:summary=self.run_saved(args)['summary']
            finally:Owner.fail_p2=False
            conditional=summary['conditional_on_continuous_measured_state_agreement']
            self.assertEqual((conditional['matched'],conditional['unknown']),(3,3))
            self.assertEqual(summary['actual_engine_decisions'],4)

    def test_external_hash_and_saved_physical_chain_mutations_rejected(self):
        for fault in ('hash','physical','chain','roster','extra'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as temp:
                args=fixture(Path(temp))
                if fault=='hash':args['expected_native_decisions_sha256']='0'*64
                elif fault=='roster':
                    native=json.loads(args['native_result_path'].read_text());native['roots'].reverse()
                    args['native_result_path'].write_text(json.dumps(native));args['expected_native_result_sha256']=trace.digest_file(args['native_result_path'])
                else:
                    rows=[json.loads(line) for line in args['native_decisions_path'].read_text().splitlines()]
                    if fault=='physical':rows[1]['native_compared_state']['battle']['sides'][0]['pokemon'][0]['hp']=12
                    elif fault=='chain':rows[1]['native_before_state_sha256']='0'*64
                    else:rows.append(deepcopy(rows[-1]))
                    args['native_decisions_path'].write_text(''.join(json.dumps(row)+'\n' for row in rows))
                    digest=trace.digest_file(args['native_decisions_path']);args['expected_native_decisions_sha256']=digest
                    native=json.loads(args['native_result_path'].read_text());native['decisions_artifact']['sha256']=digest
                    args['native_result_path'].write_text(json.dumps(native));args['expected_native_result_sha256']=trace.digest_file(args['native_result_path'])
                with self.assertRaises(ValueError):self.run_saved(args)


if __name__=='__main__':unittest.main()
