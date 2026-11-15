"""The depth-N pricer's wide, compact leaf layout: lane packing and placement, and frames equal to the blocks layout's.

leaf_layout='wide' packs only the enumerated leaf lanes into one wide iteration per call (the forward in chunks of the
blocks program's B*K rows) and writes them back to the blocks layout's dense positions. On the inert-world fakes every
frame array must equal the blocks layout's (and the oracle's), at any geometry: lane- or node-bound calls, partial calls
with dead forward chunks, several learner actors, waiting seats, terminal and error lanes, digests on and off, depth 3
(the expand ply stays blocks). Planted bugs in the wide path (lane placement off by one, swapped slots, a neighbour's
chance draw, a dropped lane) must break the equality.

What these tests do NOT cover: the fakes' forward is row-local arithmetic (each row's output depends on its own row only, in
any batch), so they check packing, placement and the per-lane plumbing, not numeric invariance. Whether the real champion
forward gives a lane the same bits at another row of its chunk (its value and its usability code, error 4) is a property of
the compiled GEMMs: it held on the GB300 frame gate under the pinned stack and not on CPU, and
scripts/depthn_wide_leaf_frame_gate.py re-checks it per stack (its CLI and gate are exercised here on the fake engine, a
leafless batch included)."""
import inspect
import textwrap
import unittest
from unittest.mock import patch

import jax
import numpy as np

from gpu_engine import device_depthn_pricer as dn
from tests import test_gpu_engine_depth2_pricer as T
from tests import test_gpu_engine_depthn_pricer as DN
from tests import test_gpu_engine_depthn_live_search as LS
from tests.test_gpu_engine_device_public_teacher import facts

SPECS=(((0,27),0,0,1),((27,21),1,1,0),((22,23),0,2,2),((24,25),1,0,2),((29,30),0,1,1),((20,6),1,2,0))   # three learner actors


def mutant(fn,old,new):
    """`fn` with one source edit, compiled in its module's namespace (a planted bug for the negative controls)."""
    src=textwrap.dedent(inspect.getsource(fn))
    if src.count(old)!=1:raise AssertionError(f'mutation site not unique in {fn.__name__}: {old!r}')
    space=dict(vars(inspect.getmodule(fn)));exec(compile(src.replace(old,new),inspect.getsourcefile(fn),'exec'),space)
    return space[fn.__name__]


class PackingTests(unittest.TestCase):
    def pack(self,group,own,opp,**kw):
        own=np.asarray(own,bool);opp=np.asarray(opp,bool);p=dn.pack_wide_leaf(group,own,opp,**kw)
        k_opp=opp.shape[1];L=kw['lanes'];per_call=np.bincount(p.lane_pos//L,minlength=p.calls)
        self.assertTrue((per_call<=L).all());self.assertTrue(((p.call_nodes>=0).sum(1)<=kw['nodes']).all())
        expect=[(i,s,x)for i in range(len(group))for s in range(own.shape[1])for x in range(k_opp)if own[i,s]and opp[i,x]]
        self.assertEqual(list(zip(p.lane_parent.tolist(),p.lane_slot_own.tolist(),p.lane_slot_opp.tolist())),expect)   # every lane once, in order
        for i in np.unique(p.lane_parent):   # a parent's lanes are contiguous in one call, its node slot holds it
            pos=p.lane_pos[p.lane_parent==i];self.assertEqual(pos.tolist(),list(range(pos[0],pos[0]+len(pos))));self.assertEqual(len(set(pos//L)),1)
            c=pos[0]//L;self.assertEqual(p.call_nodes[c,p.node_slot[i]],i);self.assertTrue((p.lane_node.reshape(-1)[pos]==p.node_slot[i]).all())
        flat=lambda x:x.reshape(-1)
        self.assertEqual(int(flat(p.lane_live).sum()),len(expect));self.assertTrue((flat(p.lane_own)[p.lane_pos]==p.lane_slot_own).all())
        self.assertTrue((flat(p.lane_opp)[p.lane_pos]==p.lane_slot_opp).all())
        self.assertTrue((flat(p.chunk_group)[p.lane_pos//kw['chunk']]==np.asarray(group)[p.lane_parent]).all())
        return p

    def test_greedy_calls_keep_a_parent_whole(self):
        live=lambda c:[True]*c+[False]*(2-c)
        p=self.pack([0,0,0,0],[live(2),live(1),live(2),live(1)],[live(2),live(2),live(1),live(1)],nodes=8,lanes=8,chunk=4)
        # 4 + 2 lanes fill call 0 to 6; the third parent's 2 lanes fit (8); the fourth's 1 lane opens call 1.
        self.assertEqual((p.calls,p.node_call.tolist(),p.node_lane.tolist()),(2,[0,0,0,1],[0,4,6,0]))
        self.assertEqual(p.chunk_group.tolist(),[[0,0],[0,-1]])
        p=self.pack([0,0,0],[live(2)]*3,[live(2)]*3,nodes=8,lanes=8,chunk=4)
        self.assertEqual((p.calls,p.node_call.tolist(),p.node_lane.tolist()),(2,[0,0,1],[0,4,0]))

    def test_node_capacity_closes_a_call(self):
        p=self.pack([0]*5,[[True,False]]*5,[[True,False]]*5,nodes=2,lanes=8,chunk=4)
        self.assertEqual((p.calls,p.node_call.tolist(),p.node_slot.tolist()),(3,[0,0,1,1,2],[0,1,0,1,0]))
        self.assertEqual(p.call_nodes.tolist(),[[0,1],[2,3],[4,-1]])

    def test_a_group_change_starts_a_forward_chunk(self):
        one=[[True,False]]
        p=self.pack([0,0,1,2,2],one*5,one*5,nodes=9,lanes=8,chunk=4)
        # group 0 at lanes 0,1; group 1 at the next chunk (4); group 2 cannot start a chunk in call 0 (8 is full): call 1.
        self.assertEqual((p.calls,p.node_call.tolist(),p.node_lane.tolist()),(2,[0,0,0,1,1],[0,1,4,0,1]))
        self.assertEqual(p.chunk_group.tolist(),[[0,1],[2,-1]])

    def test_parents_without_lanes_are_not_packed(self):
        p=self.pack([0,0,0],[[True,True],[False,False],[True,False]],[[True,False],[True,True],[False,True]],nodes=4,lanes=4,chunk=4)
        self.assertEqual((p.node_call.tolist(),p.node_lane.tolist()),([0,-1,0],[0,-1,2]))
        empty=self.pack([],np.zeros((0,3),bool),np.zeros((0,2),bool),nodes=4,lanes=12,chunk=6)
        self.assertEqual((empty.calls,empty.lane_live.shape,len(empty.lane_pos)),(0,(0,12),0))

    def test_random_packings_and_placement_roundtrip(self):
        rng=np.random.default_rng(7)
        for trial in range(40):
            P=int(rng.integers(1,60));k_own,k_opp=int(rng.integers(1,5)),int(rng.integers(1,5));K=k_own*k_opp
            chunk=K*int(rng.integers(1,4));lanes=chunk*int(rng.integers(1,5));nodes=int(rng.integers(1,12))
            group=np.sort(rng.integers(0,3,P));own=rng.random((P,k_own))<.6;opp=rng.random((P,k_opp))<.6
            p=self.pack(group,own,opp,nodes=nodes,lanes=lanes,chunk=chunk)
            flat=rng.permutation(4*50)[:P];pw,pid=flat//50,flat%50   # distinct (world, node) per parent
            fields=dict(v=np.arange(p.calls*lanes,dtype=np.float32)+1,d=np.stack([np.arange(p.calls*lanes)]*2,-1).astype(np.uint32)+1)
            out=dn.place_wide_leaf(p,fields,pw,pid,4,50,k_own,k_opp,[('v',(),np.float32),('d',(2,),np.uint32)])
            for i in range(P):
                for s in range(k_own):
                    for x in range(k_opp):
                        hit=(p.lane_parent==i)&(p.lane_slot_own==s)&(p.lane_slot_opp==x)
                        want=fields['v'][p.lane_pos[hit][0]]if hit.any()else 0.
                        self.assertEqual(out['v'][pw[i],pid[i],s,x],want,f'trial {trial}')
                        self.assertEqual(out['d'][pw[i],pid[i],s,x].tolist(),[int(want)]*2)
            self.assertEqual(int(np.count_nonzero(out['v'])),len(p.lane_pos))

    def test_fail_closed(self):
        one=np.ones((2,2),bool)
        for kw in(dict(nodes=0,lanes=8,chunk=4),dict(nodes=2,lanes=6,chunk=4),dict(nodes=2,lanes=6,chunk=3)):   # no nodes; lanes % chunk; chunk < K
            with self.assertRaises(ValueError):dn.pack_wide_leaf([0,0],one,one,**kw)
        with self.assertRaises(ValueError):dn.pack_wide_leaf([0,1,0],np.ones((3,2),bool),np.ones((3,2),bool),nodes=4,lanes=8,chunk=4)   # a group split


class FrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.root,_,cls.tables,cls.recipe=facts()

    def requests(self,specs,ex):return T.Depth2PricerTests.requests(self,specs,ex)
    def price(self,specs,**kw):return DN.DepthNPricerTests.price(self,specs,**kw)
    def check_oracle(self,*a,**kw):return DN.DepthNPricerTests.check_oracle(self,*a,**kw)
    def assert_frames_equal(self,a,b,msg=''):return DN.DepthNPricerTests.assert_frames_equal(self,a,b,msg)

    def both(self,specs,*,ex=None,wide=None,**kw):
        """(blocks frame, wide frame, wide engine): one fake executor, two fresh pricers at the same geometry."""
        ex=ex or DN.deep_executor()
        blocks=self.price(specs,ex=ex,**kw)[0];frame,_,engine,_=self.price(specs,ex=ex,leaf_layout='wide',**dict(kw,**(wide or {})))
        self.assertGreater(int(np.count_nonzero(blocks.leaf_status)),0)
        return blocks,frame,engine

    def test_depth2_wide_equals_blocks_across_geometries(self):
        for ex in(T.executor()[0],T.permuting_executor(),DN.lane_executor()):
            for (k_own,k_opp),lanes,calls,wide,digest in(((9,9),5*81,2,{},False),((9,9),81,4,{},True),((9,9),81,1,dict(wide_nodes=1),False),
                    ((2,3),40,3,dict(wide_nodes=2),True),((3,2),6,16,{},False),((9,9),2187,16,{},True)):
                blocks,frame,engine=self.both(SPECS,ex=ex,depth=2,k_own=(k_own,),k_opp=(k_opp,),block_lanes=lanes,blocks_per_call=calls,digest=digest,wide=wide)
                self.assert_frames_equal(blocks,frame,f'k=({k_own},{k_opp}) lanes={lanes} calls={calls} {wide}')
                self.assertEqual([w['status']for pair in blocks.world_reports for w in pair],[w['status']for pair in frame.world_reports for w in pair])
                t=frame.timing['plies'][-1];self.assertEqual(t['packed_lanes'],t['enumerated'])
                self.assertEqual(t['enumerated'],blocks.timing['plies'][-1]['enumerated'])

    def test_depth3_wide_leaf_after_blocks_expand_matches_blocks_and_the_oracle(self):
        specs=(((0,27),0,0,1),((27,0),1,1,0),((0,0),0,2,2))
        blocks,frame,engine=self.both(specs,depth=3,k_own=(2,3),k_opp=(3,2),block_lanes=40,blocks_per_call=4,digest=True)
        self.assert_frames_equal(blocks,frame)
        self.check_oracle(frame,self.requests(specs,DN.deep_executor()),DN.deep_executor(),specs,((2,3),(3,2)))
        kinds=[k[:2]for k in engine.programs if k[0]in('expand','leaf')]
        self.assertEqual(kinds,[('expand',6),('leaf','wide')])   # ply 2 expands on the blocks program (B=6), the leaf is wide
        self.assertNotIn('layout',frame.timing['plies'][0]);self.assertEqual(frame.timing['plies'][1]['layout'],'wide')

    def test_waiting_terminal_error_and_padding_lanes(self):
        specs=(((40,41),0,0,1),((20,21),1,1,0),((22,23),0,0,1),((24,25),1,1,0),((43,44),0,2,1),((28,27),1,1,0),((46,47),0,1,2))
        blocks,frame,_=self.both(specs,depth=3,k_own=(3,2),k_opp=(2,3),block_lanes=40,blocks_per_call=2,wide=dict(wide_nodes=3))
        self.assert_frames_equal(blocks,frame)
        self.assertTrue({1,3,4}<=set(np.unique(frame.leaf_status).tolist()))   # terminal, error and head leaves all placed
        specs=(((0,0),0,0,1),((0,27),1,1,0),((32,32),0,0,2));missing=((1,1),)   # a missing world and padding roots
        blocks=self.price(specs,depth=3,k_own=(2,2),k_opp=(2,2),missing=missing,block_lanes=20,blocks_per_call=3)[0]
        frame=self.price(specs,depth=3,k_own=(2,2),k_opp=(2,2),missing=missing,block_lanes=20,blocks_per_call=3,leaf_layout='wide')[0]
        self.assert_frames_equal(blocks,frame)

    def test_keys_timing_and_the_default(self):
        blocks,frame,engine=self.both(SPECS,depth=2,k_own=(9,),k_opp=(9,),block_lanes=81,blocks_per_call=4,digest=True)
        keys=[k for k in engine.programs if k[0]=='leaf'];self.assertEqual(keys,[('leaf','wide',20,324,81,9,9)])   # nodes = lanes // 16
        self.assertEqual([k for k in blocks.timing['programs']if k[0]=='leaf'],[('leaf',1,4,9,9)])
        self.assertTrue(any(k[0]=='gather'and k[2]==20 for k in engine.programs))
        b,w=blocks.timing['plies'][-1],frame.timing['plies'][-1]
        self.assertTrue(set(b)<set(w));self.assertNotIn('layout',b);self.assertNotIn('leaf_layout',blocks.timing);self.assertEqual(frame.timing['leaf_layout'],'wide')
        self.assertEqual((w['layout'],w['node_capacity'],w['lane_capacity'],w['forward_chunk_rows'],w['blocks_per_call']),('wide',20,324,81,4))
        self.assertEqual((w['evaluated_lanes'],w['lanes'],w['forward_chunks']),(w['calls']*324,w['calls']*324,w['calls']*4))
        self.assertEqual(w['packed_lanes'],int(np.count_nonzero(frame.leaf_status)));self.assertLessEqual(w['packed_nodes'],w['parents'])
        self.assertEqual((w['dense_blocks'],w['dense_lanes']),(b['blocks'],b['lanes']));self.assertEqual(w['forward_rows'],w['forward_chunks_live']*81)
        self.assertLess(w['calls'],b['calls']);self.assertEqual(frame.timing['evaluated_leaf_lanes'],w['lanes'])
        ex=DN.deep_executor()
        for kw in(dict(leaf_layout='rows'),dict(wide_nodes=4),dict(leaf_layout='wide',wide_nodes=0),dict(leaf_layout='wide',wide_nodes=2.)):
            with self.assertRaises(ValueError):dn.DepthNPricer(ex,None,capacity=1,depth=2,k_own=2,k_opp=2,actor_capacity=1,**kw)
        self.assertEqual(dn.DepthNPricer(ex,None,capacity=1,depth=2,k_own=2,k_opp=2,actor_capacity=1).leaf_layout,'blocks')

    def test_planted_bugs_break_the_equality(self):
        kw=dict(depth=2,k_own=(9,),k_opp=(9,),block_lanes=81,blocks_per_call=4,digest=True)
        ex=T.executor()[0];blocks=self.price(SPECS,ex=ex,**kw)[0]
        place=dn.place_wide_leaf;pack=dn.pack_wide_leaf;program=dn.make_wide_leaf_program
        def dropped(*a,**k):   # the last packed lane is never stepped
            p=pack(*a,**k);live=p.lane_live.copy().reshape(-1);live[p.lane_pos[-1]]=False
            from dataclasses import replace
            return replace(p,lane_live=live.reshape(p.lane_live.shape))
        bugs=dict(off_by_one=('place_wide_leaf',mutant(place,'np.asarray(fields[name])[pack.lane_pos]','np.asarray(fields[name])[np.minimum(pack.lane_pos+1,len(fields[name])-1)]')),
            swapped_slots=('place_wide_leaf',mutant(place,'pack.lane_slot_own,pack.lane_slot_opp]','pack.lane_slot_own[::-1],pack.lane_slot_opp[::-1]]')),
            neighbour_chance=('make_wide_leaf_program',mutant(program,'chance=jax.vmap(executor.sample)(states)',
                'chance=jax.tree.map(lambda x:j.roll(x,1,axis=0),jax.vmap(executor.sample)(states))')),
            dropped_lane=('pack_wide_leaf',dropped))
        for name,(target,bug)in bugs.items():
            with patch.object(dn,target,bug):frame=self.price(SPECS,ex=ex,leaf_layout='wide',**kw)[0]
            with self.assertRaises(AssertionError,msg=name):self.assert_frames_equal(blocks,frame)
        self.assert_frames_equal(blocks,self.price(SPECS,ex=ex,leaf_layout='wide',**kw)[0])   # unpatched again: equal


class LiveLayoutTests(unittest.TestCase):
    """make_pricers(leaf_layout=...), DepthNLiveSearch and the server flags: LiveSearchTests' harness (fake engine), not its tests."""
    LS=LS
    setUpClass=classmethod(LS.LiveSearchTests.setUpClass.__func__);make_pricers=classmethod(LS.LiveSearchTests.make_pricers.__func__)
    setUp,search,outcome,run_batch=LS.LiveSearchTests.setUp,LS.LiveSearchTests.search,LS.LiveSearchTests.outcome,LS.LiveSearchTests.run_batch

    def layout_pricers(self,buckets,**kw):
        from tests.test_gpu_engine_depthn_guard_inputs import fake_labels
        return self.LS.live.make_pricers(self.executor,self.evaluator,self.LS.PARAMS,worlds=8,buckets=buckets,block_lanes=81,blocks_per_call=4,labels=fake_labels,**kw)

    def test_wide_pricers_warm_up_serve_without_compiling_and_answer_as_blocks(self):
        LS=self.LS;outcomes={('w0',0):self.outcome(0),('w1',0):self.outcome(1),**{(f'l{i}',0):self.outcome(2+i)for i in range(3)}};answers={}
        for layout in('blocks','wide'):
            pricers=self.layout_pricers((1,2),**({}if layout=='blocks'else dict(leaf_layout='wide')));s=self.search(pricers=pricers)
            self.assertTrue(all(p.leaf_layout==layout for p in pricers.values()))
            with LS.Codex(outcomes):report=s.warmup([LS.request('w0',0),LS.request('w1',0)])
            self.assertEqual([s.missing_programs(b)for b in(1,2)],[[]]*2);self.assertTrue(all(e['warm_compiled']==0 for e in report['buckets'].values()))
            leaf=[k for k in pricers[1].programs if k[0]=='leaf'];self.assertEqual(len(leaf),1);self.assertEqual(leaf[0][1]=='wide',layout=='wide')
            rows=[]
            for n,b in((1,1),(2,2),(1,1)):
                got,_=self.run_batch(s,[dict(LS.request(f'l{i}',0),request_id=f'l{i}:{n}:{b}')for i in range(n)],outcomes)
                self.assertEqual(pricers[b].compiles,());self.assertEqual({r['bucket']for r in got},{b});rows+=got;s.reset_sessions()
            answers[layout]=rows
            self.assertEqual({k:s.pricer_identity[k]for k in('leaf_layout','wide_nodes')if k in s.pricer_identity},{}if layout=='blocks'else dict(leaf_layout='wide',wide_nodes=20))   # the resolved capacity: 4 chunks x 81 lanes // 16
        def canon(r):
            r={k:v for k,v in r.items()if k not in('timings','decision_seconds')};r['pricer']={k:v for k,v in r['pricer'].items()if k not in('leaf_layout','wide_nodes')}
            return r
        self.assertEqual([canon(r)for r in answers['blocks']],[canon(r)for r in answers['wide']])
        self.assertTrue(all(r['status']=='COMPLETE'and r['effect_guard']is not None for r in answers['wide']))

    def test_server_flags_reach_make_pricers(self):
        import io,os
        from contextlib import ExitStack,redirect_stderr
        from types import SimpleNamespace as NS
        from tests import test_depthn_live_server as TS
        from gpu_engine import constructor_worlds as cw,device_public_teacher as teacher,engine_device_observation,phase_d_protocol_constructor as pc,transition
        S=TS.S;base=['--showdown-root','s','--node-binary','n','--metadata','m','--expected-metadata-sha256','a'*64,'--vocabulary','v',
            '--expected-vocabulary-sha256','b'*64,'--port','9000','--warmup-requests','w.jsonl','--champion-archive','c.npz','--champion-archive-sha256','c'*64,
            '--champion-checkpoint-sha256','d'*64]
        made={}
        stand_ins=((S,'matmul_precision',lambda:'highest'),(jax,'devices',lambda:[NS(platform='gpu')]),(pc,'load_inputs',lambda a:('catalog','source','metadata','config')),
            (S.evaluators,'load_from_arguments',lambda a:NS(params='champion')),(pc,'ProtocolConstructor',lambda **kw:NS(**kw)),
            (teacher,'make_executor',lambda *a,**k:'executor'),(transition,'tables_from_metadata',lambda m:'tables'),
            (engine_device_observation,'compile_tables',lambda m,catalog:'observation'),(S.live,'make_pricers',lambda *a,**k:made.update(k)or'pricers'),
            (cw,'binding',lambda *a:'binding'),(S.live,'DepthNLiveSearch',lambda **kw:'search'))
        for extra,want in(([],('blocks',None)),(['--leaf-layout','wide'],('wide',None)),(['--leaf-layout','wide','--wide-nodes','512'],('wide',512))):
            with patch.dict(os.environ,TS.PINNED):args=S.options(base+extra)
            with ExitStack()as stack:
                for target,name,value in stand_ins:stack.enter_context(patch.object(target,name,side_effect=value))
                self.assertEqual(S.build(args),'search')
            self.assertEqual((made['leaf_layout'],made['wide_nodes']),want)
        for bad in(['--leaf-layout','rows'],['--wide-nodes','512'],['--leaf-layout','blocks','--wide-nodes','8'],['--leaf-layout','wide','--wide-nodes','0'],
                ['--leaf-layout','wide','--wide-nodes','-3']):   # --wide-nodes belongs to --leaf-layout wide and is positive
            with self.subTest(bad=bad),patch.dict(os.environ,TS.PINNED),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):S.options(base+bad)

    def test_frame_gate_passes_its_negative_control_and_fails_a_planted_bug(self):
        import io,json,os,tempfile
        from contextlib import redirect_stdout
        from tests import test_depthn_live_server as TS
        from scripts import depthn_wide_leaf_frame_gate as G
        LS=self.LS;s=self.search(pricers=self.layout_pricers((1,)))
        outcomes={(f'l{i}',0):self.outcome(2+i)for i in range(3)};requests=[dict(LS.request(f'l{i}',0),request_id=f'l{i}')for i in range(3)]
        blocks,wide=G.gate_pricers(s,chunks=2);self.assertEqual((blocks.leaf_layout,wide.leaf_layout,wide.blocks_per_call),('blocks','wide',2))
        quiet=lambda line:None
        with LS.Codex(outcomes):report=G.gate(s,blocks,wide,requests,min_priced=3,log=quiet)
        self.assertEqual((report['verdict'],report['tally']['priced'],report['tally']['equal'],report['reasons']),('PASS',3,3,[]));json.dumps(report,default=repr)
        self.assertGreater(report['tally']['leaves'],0)
        with LS.Codex(outcomes):short=G.gate(s,blocks,wide,requests,min_priced=4,log=quiet)   # every root equal, but under --min-priced
        self.assertEqual(short['verdict'],'FAIL');self.assertIn('under --min-priced 4',short['reasons'][0])
        # G1: roots that price without a single leaf (every turn-1 cell terminal: mode 1) run no wide call and prove nothing
        leafless={(f'z{i}',0):self.cw.made(20+i,0,(1,)*8)for i in range(3)};flat=[dict(LS.request(f'z{i}',0),request_id=f'z{i}')for i in range(3)]
        with LS.Codex(leafless):vacuous=G.gate(s,blocks,wide,flat,min_priced=1,log=quiet)
        self.assertEqual((vacuous['verdict'],vacuous['tally']['priced'],vacuous['tally']['leafless'],vacuous['tally']['leaves']),('FAIL',0,3,0))
        self.assertEqual({r['status']for r in vacuous['roots']},{'NO_LEAVES'});self.assertIn('no enumerated leaves',vacuous['reasons'])
        with LS.Codex(leafless):control=G.gate(s,blocks,wide,flat,min_priced=1,negative_control=True,log=quiet)
        self.assertEqual(control['verdict'],'FAIL')   # a negative control over leafless roots has no power either
        fresh_wide=G.gate_pricers(s,chunks=2)
        with LS.Codex(leafless):nowide=G.gate(s,*fresh_wide,flat,min_priced=0,log=quiet)   # even with no minimum: no wide program ran
        self.assertIn('no wide leaf program in the table',nowide['reasons']);self.assertEqual(nowide['verdict'],'FAIL')
        self.assertTrue(any(k.startswith("('leaf', 'wide'")and h for k,h in report['programs'].items()))   # the compiled leaf programs' HLO hashes
        with LS.Codex(outcomes):neg=G.gate(s,blocks,wide,requests,min_priced=3,negative_control=True,log=quiet)
        self.assertEqual((neg['verdict'],neg['tally']['differ']),('PASS',3));self.assertTrue(all('leaf_value'in r['differ']or'leaf_status'in r['differ']for r in neg['roots']))
        shifted=G._off_by_one(dn.place_wide_leaf)   # the same shift as a real placement bug (no control flag): the gate FAILS
        with LS.Codex(outcomes),patch.object(dn,'place_wide_leaf',shifted):bad=G.gate(s,blocks,wide,requests,min_priced=3,log=quiet)
        self.assertEqual((bad['verdict'],bad['tally']['differ']),('FAIL',3))
        # the CLI: server flags after --, exit 0 PASS / 1 FAIL / 2 bad arguments, one summary line, the JSON report
        server=['--showdown-root','s','--node-binary','n','--metadata','m','--expected-metadata-sha256','a'*64,'--vocabulary','v',
            '--expected-vocabulary-sha256','b'*64,'--port','9000','--warmup-requests','w.jsonl','--champion-archive','c.npz','--champion-archive-sha256','c'*64,
            '--champion-checkpoint-sha256','d'*64,'--block-lanes','81','--blocks-per-call','4']
        keys=('battle_id','owner','stream_id','decision_index','generation','prefix_sha256','lines','own_request_line')
        with tempfile.TemporaryDirectory()as tmp:
            roots=os.path.join(tmp,'roots.jsonl')
            with open(roots,'w')as f:
                for r in requests:f.write(json.dumps({k:r[k]for k in keys})+'\n')
            def cli(name,*extra,bug=None):
                out=io.StringIO();argv=['--roots',roots,'--rung','DN2W8','--out',os.path.join(tmp,name),'--max-roots','2','--min-priced','2',*extra,'--']+server
                with patch.dict(os.environ,TS.PINNED),patch.object(TS.S,'build',return_value=s),LS.Codex(outcomes),redirect_stdout(out):
                    if bug is None:rc=G.main(argv)
                    else:
                        with patch.object(dn,'place_wide_leaf',bug):rc=G.main(argv)
                return rc,out.getvalue().strip().splitlines()[-1],json.load(open(os.path.join(tmp,name)))
            rc,line,rep=cli('pass.json')
            self.assertEqual(rc,0);self.assertTrue(line.startswith('WIDE_LEAF_FRAME_GATE PASS rung=DN2W8 chunks=4 roots=2 priced=2'),line)
            self.assertEqual((rep['tally']['roots'],rep['chunks'],rep['stack']['platform']),(2,4,'cpu'));self.assertIn('jax',rep['stack'])
            self.assertTrue({'xla_flags','compilation_cache_dir','autotune_cache_dir','tree_head'}<=set(rep['stack']));self.assertEqual(rep['tree_commit_label'],s.tree_commit)
            rc,line,rep=cli('neg.json','--negative-control');self.assertEqual(rc,0);self.assertIn('PASS (negative control)',line)
            rc,line,rep=cli('fail.json',bug=shifted);self.assertEqual(rc,1);self.assertIn('FAIL',line)
            # N4: a setup error (the search cannot be built) is exit 3, not a FAIL
            out=io.StringIO()
            with patch.dict(os.environ,TS.PINNED),patch.object(TS.S,'build',side_effect=RuntimeError('no GPU')),redirect_stdout(out):
                rc=G.main(['--roots',roots,'--rung','DN2W8','--out',os.path.join(tmp,'setup.json'),'--']+server)
            self.assertEqual(rc,3);self.assertIn('SETUP_ERROR build: RuntimeError: no GPU',out.getvalue());self.assertFalse(os.path.exists(os.path.join(tmp,'setup.json')))
            for bad in(['--chunks','0'],['--max-roots','0'],['--min-priced','0'],['--out',os.path.join(tmp,'missing','x.json')]):
                with self.subTest(bad=bad),redirect_stdout(io.StringIO()),__import__('contextlib').redirect_stderr(io.StringIO()),self.assertRaises(SystemExit)as e:
                    G.main(['--roots',roots,'--rung','DN2W8','--out',os.path.join(tmp,'x.json'),*bad,'--']+server)
                self.assertEqual(e.exception.code,2)


if __name__=='__main__':unittest.main()
