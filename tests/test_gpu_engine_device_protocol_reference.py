from types import SimpleNamespace
import json
import unittest
import numpy as np
from gpu_engine.device_protocol_reference import _RetainEncoded,Reference


class DeviceProtocolReferenceTests(unittest.TestCase):
    def fixture(self):
        catalog=object();state=object();result=object()
        functions=SimpleNamespace(catalog=catalog,step=object(),
            encode_step=lambda s,o,side:result,
            require_catalog=lambda c:None if c is catalog else (_ for _ in ()).throw(ValueError('namespace')))
        capture=_RetainEncoded(functions)
        class Backend:
            feeds=[]
            def feed(self,lines):
                if any(line.startswith('|UNSUPPORTED|')for line in lines):
                    raise ValueError('wire refused kind=UNSUPPORTED offset=0')
                self.feeds.append(lines)
            def encode(self,request,*,owner_side):
                return capture.encode_step(state,request,np.asarray([owner_side],np.int32))
        out=SimpleNamespace(numeric_features=np.zeros((1,29,132),np.float32),
            categorical_ids=np.zeros((1,29,41),np.int32),token_type_ids=np.zeros((1,29),np.int32),
            attention_mask=np.ones((1,29),bool),legal_action_mask=np.ones((1,9),bool),
            switch_targets=np.arange(5,dtype=np.int32)[None],admitted=np.asarray([True]))
        calls=[]
        def append(encoded,s,side):
            self.assertIs(encoded,result);self.assertIs(s,state);calls.append(1);return out
        backend=Backend();return Reference(backend,capture,append,0),capture,calls

    @staticmethod
    def request(owner='p1'):
        return '|request|'+json.dumps(dict(side=dict(id=owner)))

    def test_same_original_encode_result_appended_once(self):
        ref,capture,calls=self.fixture()
        self.assertIsNone(ref.feed(('|turn|10',self.request())))
        view,error=ref.view();self.assertIsNone(error);self.assertEqual(view['numeric_features'].shape,(29,132))
        self.assertEqual(capture.calls,1);self.assertEqual(len(calls),1);self.assertTrue(view['admitted'])

    def test_unsupported_prefix_stays_nonfatal_unknown_and_retains_kind_counts(self):
        ref,capture,calls=self.fixture()
        failure=ref.feed(('|UNSUPPORTED|actual',self.request()))
        self.assertEqual(failure['kind'],'UNSUPPORTED')
        ref.feed(('|move|p1a: X|Surf|p2a: Y',self.request()))
        view,error=ref.view();self.assertIsNone(view);self.assertEqual(error['kind'],'UNSUPPORTED')
        self.assertEqual(ref.metadata()['raw_line_counts']['move'],1)
        self.assertEqual(ref.metadata()['raw_line_counts']['UNSUPPORTED'],1)
        self.assertEqual(capture.calls,0);self.assertEqual(calls,[])

    def test_wrong_owner_or_missing_request_never_scores(self):
        ref,capture,calls=self.fixture()
        self.assertEqual(ref.view()[1]['reason'],'MISSING_OWN_REQUEST')
        self.assertEqual(ref.feed((self.request('p2'),))['kind'],'request')
        self.assertIsNone(ref.view()[0]);self.assertEqual(capture.calls,0)


if __name__=='__main__':unittest.main()
