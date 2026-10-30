import importlib.util
import copy
import json
import unittest
import numpy as np
from pokezero.mirror_value_probe import digest
from pokezero.recorded_encoder_runner import compare_battle
from pokezero.recorded_encoder_parity import ParityTally,SHAPES
import test_recorded_encoder_inventory as inventory_fixtures


class RecordedRunnerTests(unittest.TestCase):
    def rows(self):
        battle={'battle_id':'runner-authored','seed':321}
        rows=inventory_fixtures.RecordedEncoderInventoryTests().rows(battle);first=rows[0]
        second=copy.deepcopy(first);second['decision_round_index']=1
        second['requested_players']=['p1'];second['paired']=False;second['observations'].pop('p2');second['actions'].pop('p2')
        second['requests']['p2']['wait']=True
        delta=['|','|request|'+json.dumps(second['requests']['p2'])]
        second.update(protocol_start=first['protocol_end'],protocol_end=first['protocol_end']+2,
            protocol_delta=delta,public_prefix_sha256=digest(first['protocol_delta']+delta))
        second['root_id']=digest([battle['battle_id'],1,second['public_prefix_sha256'],second['requests']])
        end=dict(event='end',protocol_start=second['protocol_end'],protocol_end=second['protocol_end']+1,protocol_delta=['|win|p1'])
        return battle,[first,second,end]

    def adapter(self,rows):
        calls=[];references=[row['observations'][seat] for row in rows if not row.get('event') for seat in row['requested_players']]
        class AuthoredAdapter:
            index=0
            def reset(self,**kwargs):calls.append(('reset',kwargs))
            def feed(self,lines):
                self.assert_no_private(lines);calls.append(('feed',lines))
            def assert_no_private(self,lines):
                if any(line.startswith('|request|') for line in lines):raise AssertionError('private line reached public fold')
            def encode(self,request,*,owner_side):
                if request['side']['id']!=f'p{owner_side+1}':raise AssertionError('foreign request')
                calls.append(('encode',request['side']['id']));ref=references[self.index];self.index+=1
                return {k:np.array(ref[k],np.float32 if k=='numeric_features' else np.bool_ if k.endswith('mask') else np.int32) for k in SHAPES}
            def forward(self,observation):
                if set(observation)!=set(SHAPES):raise AssertionError('metadata reached model')
                calls.append(('forward',None));return dict(policy_logits=np.zeros(9,np.float32),value=np.float32(0))
        return AuthoredAdapter(),calls

    def test_one_shared_feed_then_own_views_including_asymmetric_boundary(self):
        battle,rows=self.rows();adapter,calls=self.adapter(rows);tally=ParityTally(expected_views=3)
        result=compare_battle(iter(rows),battle=battle,stream_id=1,adapter=adapter,tally=tally,with_forward=True)
        self.assertEqual((result['boundaries'],result['views']),(2,3))
        self.assertEqual([x[1] for x in calls if x[0]=='encode'],['p1','p2','p1'])
        self.assertEqual(len([x for x in calls if x[0]=='feed']),2)
        self.assertEqual(len([x for x in calls if x[0]=='forward']),6)
        self.assertTrue(result['terminal_validated_not_encoded']);self.assertTrue(tally.report()['model_forward_complete'])
        self.assertFalse(tally.report()['public_wire_qualified'])

    def test_missing_terminal_late_corruption_and_false_forward_mode_do_not_complete(self):
        battle,rows=self.rows()
        for bad in (rows[:-1],[*rows,rows[0]],rows[:1]+[dict(rows[1],protocol_start=0),rows[2]]):
            adapter,_=self.adapter(rows);tally=ParityTally(expected_views=3)
            with self.assertRaises(ValueError):compare_battle(iter(bad),battle=battle,stream_id=1,adapter=adapter,tally=tally,with_forward=False)
            self.assertFalse(tally.report()['recorded_forward_numeric_criteria_met'])
        adapter,_=self.adapter(rows);tally=ParityTally(expected_views=3)
        with self.assertRaises(ValueError):compare_battle(iter(rows),battle=battle,stream_id=1,adapter=adapter,tally=tally,with_forward=1)
        self.assertEqual(tally.report()['views'],0)


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class ActualJaxAdapterTests(unittest.TestCase):
    def test_owning_stream_hp_visibility_recovers_percentage_toxic_before_encode(self):
        import numpy as np
        from pokezero import showdown
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        from test_gpu_engine_v4_encoder import V4EncoderTests
        V4EncoderTests.setUpClass();f=V4EncoderTests()
        lines=(*f.public.prefix,
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100 tox',
            '|switch|p2a: Gardevoir|Gardevoir, L80|100/100 tox',
            '|-damage|p1a: Gardevoir|95/100 tox|[from] psn',
            '|-damage|p2a: Gardevoir|95/100 tox|[from] psn','|turn|1')
        for owner in (0,1):
            adapter=JaxRecordedAdapter(f.catalog,owner_side=owner)
            adapter.reset(stream_id=1,battle_id='owning-toxic')
            adapter.feed(lines)
            parser=showdown._ReplayParser('owning-toxic',complete_prefix=True,
                hp_visibility={f'p{side+1}':'exact' if side==owner else 'percentage' for side in (0,1)})
            for line in lines:parser._feed_line(line)
            toxic=adapter.state.context.toxic
            self.assertEqual(np.asarray(toxic.visibility).tolist(),[[1 if side==owner else 2 for side in (0,1)]])
            self.assertEqual(int(toxic.stage[0,1-owner]),2)
            self.assertFalse(bool(toxic.known[0,owner]))
            for side in (0,1):
                self.assertEqual(int(toxic.stage[0,side]),parser.toxic_stage[f'p{side+1}'])
                self.assertEqual(bool(toxic.known[0,side]),parser.toxic_stage_known[f'p{side+1}'])
            with self.assertRaisesRegex(ValueError,'request owner differs'):
                adapter.encode({},owner_side=1-owner)
        for bad in (True,-1,2):
            with self.assertRaises(ValueError):JaxRecordedAdapter(f.catalog,owner_side=bad)
        for bad in (np.array([True]),np.array([2],np.int32),np.array([[0]],np.int32)):
            with self.assertRaises(ValueError):
                f.e.initialize(f.catalog,stream_ids=f.j.array([1],f.j.int32),owner_sides=bad)

    def test_actual_adapter_own_requests_and_late_refusal_leave_public_boundary_atomic(self):
        from test_gpu_engine_v4_encoder import V4EncoderTests
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        V4EncoderTests.setUpClass();f=V4EncoderTests();adapter=JaxRecordedAdapter(f.catalog)
        adapter.reset(stream_id=1,battle_id='actual-authored');adapter.feed(tuple(f.public.prefix))
        for owner in (0,1):
            actual=adapter.encode(f.request(owner),owner_side=owner)
            self.assertEqual({k:v.shape for k,v in actual.items()},SHAPES)
        encode=adapter.encode_step
        for missing_column in (True,False):
            def invalid(state,own,side):
                result=encode(state,own,side);obs=result.observation
                if missing_column:obs=obs._replace(numeric_written=obs.numeric_written.at[:,:,131].set(False))
                else:obs=obs._replace(numeric_written=obs.numeric_written.at[0,0,0].set(False),
                    numeric_features=obs.numeric_features.at[0,0,0].set(1))
                return result._replace(observation=obs)
            adapter.encode_step=invalid
            with self.assertRaisesRegex(ValueError,'incomplete or refused'):
                adapter.encode(f.request(0),owner_side=0)
        adapter.encode_step=encode
        before=adapter.state;host=adapter.host
        with self.assertRaisesRegex(ValueError,'wire refused kind=UNSUPPORTED'):
            adapter.feed(('|turn|1','|UNSUPPORTED|literal'))
        self.assertIs(adapter.state,before);self.assertEqual(adapter.host,host)
        # A valid whole frame before a late device failure must not commit.
        original=adapter.step;count=0
        def injected(state,chunk):
            nonlocal count
            count+=1
            if count==2:return state,f.j.array([1],f.j.int32)
            return original(state,chunk)
        adapter.step=injected
        with self.assertRaisesRegex(ValueError,'JAX public fold refused'):
            adapter.feed(tuple(['|']*20))
        self.assertGreaterEqual(count,2);self.assertIs(adapter.state,before);self.assertEqual(adapter.host,host)
        with self.assertRaises(ValueError):adapter.reset(stream_id=1,battle_id='another')
        with self.assertRaises(ValueError):adapter.reset(stream_id=2,battle_id='actual-authored')
