"""--reply-below-root: the reply model (br_fp) as the pricer's second bank entry for every ply below the root.

Off (the default) nothing changes: one bank entry, no child actor, pack_turn1's data and compile key as before. On, build()
loads the reply archive once, binds (champion, reply) and hands DepthNLiveSearch child_opponent_actor=1; the root prior stays
br_fp's host-side substitution, so the only difference from a flag-off search is the opponent prior at the plies below the root
(the routing itself is tested on frames in tests.test_gpu_engine_depthn_pricer)."""
from contextlib import ExitStack,redirect_stderr
import io
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import device_depth2_pricer as d2,device_public_teacher_batch as batch
from scripts import depthn_live_server as S

PINNED={'JAX_DEFAULT_MATMUL_PRECISION':'highest'}
BASE=['--showdown-root','s','--node-binary','n','--metadata','m','--expected-metadata-sha256','a'*64,'--vocabulary','v',
    '--expected-vocabulary-sha256','b'*64,'--port','9000','--warmup-requests','w.jsonl','--champion-archive','c.npz','--champion-archive-sha256','c'*64,
    '--champion-checkpoint-sha256','d'*64,'--tree-commit','t'*40]
REPLY=['--reply-archive','r.npz','--reply-archive-sha256','e'*64,'--reply-checkpoint-sha256','f'*64,'--peer-view-b']


def request(learner,opponent,child=None):
    return batch.Request(plans=None,root_logits=(0.,)*9,root_legal=(True,)*9,root_inputs={},learner_actor=learner,opponent_actor=opponent,
        child_opponent_actor=child)


class HelperTests(unittest.TestCase):
    def test_request_defaults_to_the_root_opponent_actor(self):
        r=batch.Request(None,(0.,)*9,(True,)*9,{},0,1)   # the positional construction every caller used before the field
        self.assertIsNone(r.child_opponent_actor);self.assertEqual(d2.child_actor(r),1)
        self.assertEqual(d2.child_actor(request(0,0,1)),1);self.assertEqual(d2.child_actor(request(0,1,0)),0)
        self.assertFalse(d2.routes_children([request(0,0),request(0,1),request(0,1,1)]))   # equal to the root's: nothing routes
        self.assertTrue(d2.routes_children([request(0,0),request(0,0,1)]))

    def test_the_child_actor_is_validated_against_the_bank(self):
        for bad in(-1,2,'1',1.0):
            with self.assertRaisesRegex(ValueError,'child opponent actor'):d2.check_child_actor(request(0,0,bad),('a','b'))
        d2.check_child_actor(request(0,0,1),('a','b'));d2.check_child_actor(request(0,0),('a',))

    def test_actor_rows_and_the_key_term(self):
        rows=d2.actor_rows([0,0,1,1,0,0],capacity=4,params=('a','b'))   # 3 requests x 2 worlds; actor 1 holds worlds 2 and 3
        self.assertEqual([len(r)for r in rows],[8,2]);np.testing.assert_array_equal(rows[0],[0,1,4,5,8,8,8,8]);np.testing.assert_array_equal(rows[1],[2,3])
        fixed=d2.actor_rows([0,0,1,1,0,0],capacity=4,params=('a','b'),fixed_capacity=True)
        self.assertEqual([len(r)for r in fixed],[8,8]);np.testing.assert_array_equal(fixed[1],[2,3]+[8]*6)
        self.assertEqual(d2.child_rows_key(dict(peer_rows=rows)),());self.assertEqual(d2.child_rows_key(dict(child_peer_rows=rows)),(('child_peer_rows',(8,2)),))


class ServerTests(unittest.TestCase):
    def options(self,extra):
        with patch.dict(os.environ,PINNED):return S.options(BASE+extra)

    def test_the_flag_needs_the_reply_archive(self):
        with self.assertRaises(SystemExit),redirect_stderr(io.StringIO()):self.options(['--reply-below-root'])
        self.assertIs(self.options(REPLY+['--reply-below-root']).reply_below_root,True);self.assertIs(self.options(REPLY).reply_below_root,False)

    def build(self,extra):
        """build() on stand-ins (as test_depthn_live_server's build test): what make_pricers and DepthNLiveSearch were given."""
        import jax
        from gpu_engine import constructor_worlds as cw,device_public_teacher as teacher,engine_device_observation,phase_d_protocol_constructor as pc,transition
        from gpu_engine import depthn_reply_prior as rp
        made={};pricers={}
        stand_ins=((S,'matmul_precision',lambda:'highest'),(jax,'devices',lambda:[NS(platform='gpu')]),(pc,'load_inputs',lambda a:('catalog','source','metadata','config')),
            (S.evaluators,'load_from_arguments',lambda a:NS(params='champion')),(pc,'ProtocolConstructor',lambda **kw:NS(**kw)),
            (teacher,'make_executor',lambda *a,**k:'executor'),(transition,'tables_from_metadata',lambda m:'tables'),
            (engine_device_observation,'compile_tables',lambda m,catalog:'observation'),
            (S.live,'make_pricers',lambda executor,forward,params,**k:pricers.update(params=params)or'pricers'),
            (cw,'binding',lambda *a:'binding'),(S.live,'DepthNLiveSearch',lambda **kw:made.update(kw)or'search'),
            (S.evaluators,'load',lambda path,**kw:NS(params='reply',meta=dict(extra=dict(view='b')))),
            (rp,'ReplyPrior',lambda executor,model,mode,identity:('reply-prior',executor,model.params,mode,identity)))
        args=self.options(extra)
        with ExitStack()as stack:
            for target,name,value in stand_ins:stack.enter_context(patch.object(target,name,side_effect=value))
            self.assertEqual(S.build(args),'search')
        return args,made,pricers

    def test_off_builds_exactly_as_before(self):
        args,made,pricers=self.build(REPLY)
        self.assertEqual(pricers['params'],('champion',));self.assertNotIn('child_opponent_actor',made);self.assertEqual(made['reply'][0],'reply-prior')
        search=NS(tree_commit='t',rung='DN2W16',max_requests=1,min_price_seconds=1.,session_cap=256,refuse_classes=(),pricer_identity={})
        self.assertNotIn('reply_below_root',S.header(args,search))

    def test_on_binds_the_reply_model_as_bank_entry_one_below_the_root(self):
        args,made,pricers=self.build(REPLY+['--reply-below-root'])
        self.assertEqual(pricers['params'],('champion','reply'));self.assertEqual(made['child_opponent_actor'],1)
        self.assertEqual(made['reply'][0],'reply-prior');self.assertEqual(made['reply'][2],'reply')   # the root's br_fp substitution is unchanged
        search=NS(tree_commit='t',rung='DN2W16',max_requests=1,min_price_seconds=1.,session_cap=256,refuse_classes=(),pricer_identity={},child_opponent_actor=1)
        self.assertIs(S.header(args,search)['reply_below_root'],True);self.assertEqual(S.header(args,search)['root_operator'],'br_fp')


if __name__=='__main__':unittest.main()
