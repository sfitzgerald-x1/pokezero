"""DepthNLiveSearch on the fake engine: Codex's real ProtocolConstructor.prepare (its heavy collaborators faked), the real
constructor_worlds path and DepthNPricer; the response contract, causes, buckets, sessions and warmup; and the lead's
equivalence of action_mean_q with search_hedge.averaging."""
import collections
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from pokezero import depthn_effect_guard as G,depthn_live_actor as actor,depthn_root_operator as root_ops,v5_gumbel as original
from pokezero.phase_d_search_server import SCHEMA,encode
from gpu_engine import constructor_worlds as cw,depthn_backups as backups,depthn_live_search as live,search_hedge as hedge
from gpu_engine import phase_d_protocol_constructor as pc
from gpu_engine.phase_d_live_worlds import Refusal
from tests import test_gpu_engine_constructor_worlds as CW

PARAMS=CW.PARAMS[:1]
U=2.**-53
KEYS={'schema','request_id','rung','archive_sha256','status','action_index','q_values','refusal','cause','requested_worlds','constructed_worlds',
    'decision_seconds','root_operator','policy_target','root_value','action_policy_target','block_causes','world_refusals','timings','batch_size',
    'bucket','tree_commit','pricer','root_operator_impl','completed_q','action_mean_q','mean_q_agrees','reduction_note','root_classes','root_class',
    'root_class_source','root_class_crosscheck','root_view_source','b2_view','effect_guard'}
TREE='0123456789abcdef'*2+'01234567'
TARGET='TARGET_ARGMAX'
OPERATOR_KEYS=('root_operator','root_operator_impl','target_fallback')   # what a TARGET_ARGMAX response adds or names differently
NOISE='NOISE_GATED_K2'
NOISE_KEYS=('root_operator','root_operator_impl','noise_gate')   # what a NOISE_GATED_K2 response adds or names differently
# sha256 of default_corpus's lines, computed by this very function (its source extracted from this file) on the 3bd60120 tree
# (screen-r3's, before any root operator): the default operator's decisions and responses, byte for byte
DEFAULT_CORPUS_3BD60120=dict(decide='4a7327b026c7833de9f3ab5bdc73744eb9b158342d57ff22c8588b2d8d004020',
    responses='f9f7ba79992d575aeb1a3fea6a91721ee2a25135c4caa481f180f2cfff6e829e')


def wire(value):
    """A response field as the wire (and every reader of the telemetry or the re-pricer rows) carries it."""
    return json.loads(json.dumps(value))


def favour(action):
    """latency.compose with its reported policy_target replaced by one that puts .9 on `action` (when legal): the target's
    argmax, so TARGET_ARGMAX's choice, is `action` whatever averaging plays."""
    from gpu_engine import d3_search_latency as latency
    compose=latency.compose
    def composed(improvements,logits,legal,original):
        out=compose(improvements,logits,legal,original)
        return dict(out,policy_target=tuple((.9 if a==action else .05)if legal[a]else 0. for a in range(9)))
    return patch.object(latency,'compose',side_effect=composed)


def still_clocks(stack):
    """Every stage timing 0 (live and constructor_worlds read time.perf_counter), so whole responses compare."""
    for module in(live,cw):stack.enter_context(patch.object(module,'time',NS(perf_counter=lambda:0.,monotonic=lambda:0.)))


def default_corpus(test):
    """{decide: [...], responses: [...]}: JSON lines of the DEFAULT operator's output on fixed inputs: live.decide (no operator
    argument) on 720 seeded random roots (W 2, 4, 8, 16; every value mode, random and all-zero root logits), and, through the
    LiveSearchTests harness `test`, process_many's responses (stage clocks still) for three priced roots and a refused one at
    W 8 and 16. Uses only what the 3bd60120 tree has (DEFAULT_CORPUS_3BD60120)."""
    rng=np.random.default_rng(20261110);decided=[]
    def random_world(legal,mode):
        requested=rng.random()<.8;opponent=[b for b in range(9)if rng.random()<.5]if requested else None
        if requested and not opponent:opponent=[int(rng.integers(9))]
        value={'f64':lambda:float(rng.uniform(-1,1)),'f32':lambda:float(np.float32(rng.uniform(-1,1))),
            'terminal':lambda:float(rng.choice([-1.,0.,1.])),'ties':lambda:float(rng.choice([-1.,0.,1.]))}[mode]
        logits=(0.,)*9 if mode=='ties'else[float(x)for x in rng.normal(0,2,9)]
        return world({a:{b:value()for b in(opponent or[0])}for a in range(9)if legal[a]},opponent=opponent,logits=logits)
    for mode,cases in(('f64',40),('f32',40),('terminal',40),('ties',60)):
        for W in(2,4,8,16):
            for _ in range(cases):
                legal=tuple(bool(x)for x in rng.random(9)<.6)
                if not any(legal):legal=(True,)+legal[1:]
                logits=(0.,)*9 if mode=='ties'else tuple(float(x)for x in rng.normal(0,1,9))
                prices=[random_world(legal,mode)for _ in range(W)]
                decided.append(json.dumps(live.decide([tuple(prices[2*k:2*k+2])for k in range(W//2)],logits,legal)))
    responses=[]
    for W in(8,16):
        reqs=[request(f'b{i}',0,stream=10+i,rung=f'DN2W{W}')for i in range(3)]+[request('r',0,rung=f'DN2W{W}')]
        outcomes={**{(f'b{i}',0):test.outcome(i,W)for i in range(3)},('r',0):Refusal('SEARCH_PREFIX_OR_REQUEST_MISMATCH')}
        with ExitStack()as stack:
            for module in(live,cw):stack.enter_context(patch.object(module,'time',NS(perf_counter=lambda:0.,monotonic=lambda:0.)))
            rows,_=test.run_batch(test.search(W,clock=lambda:0.),reqs,outcomes)
        responses+=[json.dumps(r)for r in rows]
    return dict(decide=decided,responses=responses)


def prefix(battle,decision,*,owner='p1',last=None,before=()):
    """A player-stream prefix whose decision d+1 extends decision d (the incremental fold's successor rule): literal own
    requests (Codex's _packet parses them), `last` merged into the final one and `before` inserted ahead of it."""
    lines=['|init|battle',f'|title|{battle}']
    for d in range(decision+1):
        own=dict(rqid=d,side=dict(id=owner,name=owner,pokemon=[dict(ident=f'{owner}a: Snorlax',details='Snorlax, L80',condition='100/100',active=True)]))
        lines+=[f'|turn|{d+1}',*(before if d==decision else()),'|request|'+json.dumps(dict(own,**(last or{})if d==decision else{}),separators=(',',':'))]
    return lines


def payload(**sides):
    """A public materialization payload (capture_root's) with benign sides, `sides` overriding one side's dict."""
    base={side:dict(pokemon=[dict(species='Snorlax',condition='100/100',active=True)],volatiles=[])for side in('p1','p2')}
    base.update(sides);return dict(turn=1,sides=base,wishSetTurns={})


def request(battle,decision,*,owner='p1',stream=7,rung='DN2W8',budget=3.,last=None,before=(),**over):
    lines=prefix(battle,decision,owner=owner,last=last,before=before)
    out=dict(schema=SCHEMA,request_id=f'{battle}:{decision}',battle_id=battle,owner=owner,stream_id=stream,decision_index=decision,generation=1,
        prefix_sha256=hashlib.sha256(''.join(x+'\n'for x in lines).encode()).hexdigest(),lines=lines,own_request_line=lines[-1],rung=rung,budget_seconds=budget)
    out.update(over);return out


class Codex:
    """Codex's ProtocolConstructor.prepare runs unchanged (key set, schema, address, prefix hash, `seen`, `sessions`); only its
    heavy collaborators are faked: the B2 adapter (records its resets and fed lines), the public-root capture (raises
    outcomes[(battle, decision)] when that is an exception), from_b2 and the world sampler (that outcome's made() root and
    WorldSet) and B2's legal mask (the outcome's). The capture's public payload is payloads[(battle, decision)] (default
    payload()): the live search's classification reads it through the same capture_root. The own request is Codex's real
    _packet."""
    def __init__(self,outcomes,payloads=None):self.outcomes=outcomes;self.payloads=payloads or{};self.adapters=[];self.captures=[];self.current=None

    def __enter__(self):
        harness=self
        class Adapter:
            def __init__(self,catalog,owner_side,compiled_functions):self.fed=[];self.resets=[];harness.adapters.append(self)
            def reset(self,*,stream_id,battle_id):self.resets.append((stream_id,battle_id))
            def feed(self,lines):self.fed.extend(lines)
            def encode(self,packet,*,owner_side):return dict(legal_action_mask=harness.current[2])
        def capture(*,config,contract,lines,address,battle_id,seed):
            harness.captures.append((battle_id,address.frame));out=harness.outcomes[(battle_id,address.frame)]
            if isinstance(out,BaseException):raise out
            harness.current=out;return NS(address=out[0].address,materialization_payload=harness.payloads.get((battle_id,address.frame))or payload())
        self.stack=ExitStack()
        # Batch 2's prepare also reads the protocol tail's facts and passes defer_sleep / protocol_facts to from_b2: fake the
        # first (its shape: phase_d_tail.facts) and accept the second, so the outcome's root and worlds come through unchanged.
        tail=getattr(pc.protocol,'tail',None)
        for target,name,value in((pc,'JaxRecordedAdapter',Adapter),(pc,'capture_root',capture),
                (pc.worlds,'from_b2',lambda adapter,packet,source,metadata,*,address,**kw:harness.current[0]),
                (pc.protocol,'sample',lambda public,detached,**kw:harness.current[1]),
                *(()if tail is None else((tail,'facts',lambda *a,**k:dict(replacement=None,fresh_substitutes={},wish_set_turns={},turn=0)),))):
            self.stack.enter_context(patch.object(target,name,value))
        return self

    def __exit__(self,*exc):return self.stack.__exit__(*exc)


class LiveSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CW.ConstructorWorldsTests.setUpClass();cls.cw=CW.ConstructorWorldsTests('test_both_root_moves_are_reported_and_can_differ')
        cls.executor=CW.carry_executor();cls.evaluator=CW.Evaluator()
        cls.pricers={8:cls.make_pricers(8,(1,2,4)),16:cls.make_pricers(16,(1,))}

    @classmethod
    def make_pricers(cls,W,buckets,executor=None):
        from tests.test_gpu_engine_depthn_guard_inputs import fake_labels   # the fake engine's states: the effect guard's label gather
        return live.make_pricers(executor or cls.executor,cls.evaluator,PARAMS,worlds=W,buckets=buckets,block_lanes=81,blocks_per_call=4,labels=fake_labels)

    def setUp(self):
        # The fake engine's carries are not the device tracker carry: prepare_roots and build keep them as built (the carry
        # completion is tests/test_gpu_engine_constructor_worlds_carry's subject).
        keep=patch.object(cw,'world_completer',return_value=lambda world,owner,**kw:world);keep.start();self.addCleanup(keep.stop)

    def search(self,W=8,*,pricers=None,saved_root_admission=False,**kw):
        kw.setdefault('tree_commit',TREE)
        constructor=self.cw.constructor(W,saved_root_admission=saved_root_admission,rung=f'DN2W{W}')
        return live.DepthNLiveSearch(constructor=constructor,pricers=pricers or self.pricers[W],params=PARAMS,binding=self.cw.binding,recipe=self.cw.recipe,**kw)

    def outcome(self,i,W=8,**kw):return self.cw.made(i,0,self.cw.modes(i,W),**kw)

    def run_batch(self,search,requests,outcomes):
        with Codex(outcomes)as codex:rows=search.process_many(requests)
        return rows,codex

    # the contract --------------------------------------------------------------------------------------------------------------
    def test_responses_name_the_root_priors_source_and_carry_b2_view(self):
        made,_=self.cw.wide_root(0,8)
        s=self.search(pricers=self.make_pricers(8,(1,),executor=CW.wide_executor()))
        bad=self.cw.made(1,0,self.cw.modes(1,8),b2=made[0].view['_b2_root_inputs'])   # the narrow fake encoder cannot take B2's tokens
        rows,_=self.run_batch(s,[request('b2',0)],{('b2',0):made})
        row=rows[0];encode(row);self.assertEqual(set(row),KEYS)
        self.assertEqual((row['status'],row['root_view_source']),('COMPLETE','b2'))
        self.assertEqual(row['b2_view'],dict(equal=False,keys=dict(numeric_features=False,categorical_ids=True,token_type_ids=True,attention_mask=True),
            numeric_max_abs_diff=.5,error=None))
        rows,_=self.run_batch(self.search(),[request('bad',0)],{('bad',0):bad})
        row=rows[0];encode(row)
        self.assertEqual((row['status'],row['cause'],row['root_view_source'],row['b2_view']['equal']),('REFUSED','depthn_refused','b2',None))
        self.assertTrue(row['refusal'].startswith('ROOT_B2_INPUTS_UNUSABLE'));self.assertIn('KeyError',row['b2_view']['error'])
        self.assertEqual(live.wire_b2_view(dict(equal=True,keys={'numeric_features':np.bool_(True)},numeric_max_abs_diff=float('nan'))),
            dict(equal=True,keys={'numeric_features':True},numeric_max_abs_diff=None,error=None))   # never a NaN on the wire
        self.assertIsNone(live.wire_b2_view(None))

    def test_complete_responses_follow_the_contract_and_the_direct_pipeline(self):
        reqs=[request(f'b{i}',0,stream=10+i)for i in range(3)];outcomes={(f'b{i}',0):self.outcome(i)for i in range(3)}
        s=self.search();rows,codex=self.run_batch(s,reqs,outcomes)
        self.assertEqual([r['request_id']for r in rows],[r['request_id']for r in reqs]);self.assertEqual(len(set(codex.captures)),3)
        legal=CW.T.ROOT_LEGAL
        for row in rows:
            self.assertEqual(set(row),KEYS);encode(row);json.loads(json.dumps(row,allow_nan=False))
            self.assertEqual((row['root_view_source'],row['b2_view']),('world0',None))   # no B2 encoding on these roots: world 0's view
            self.assertEqual((row['schema'],row['rung'],row['status'],row['cause'],row['refusal'],row['root_operator'],row['root_operator_impl'],row['reduction_note']),
                (SCHEMA,'DN2W8','COMPLETE',None,None,'ARGMAX_MEAN_COMPLETED_Q','search_hedge.averaging',None))
            self.assertEqual(row['mean_q_agrees'],row['action_mean_q']==row['action_index'])
            self.assertEqual((row['root_classes'],row['root_class'],row['root_class_source']),([],'other','protocol_facts'))
            self.assertEqual((row['archive_sha256'],row['tree_commit'],row['requested_worlds'],row['constructed_worlds'],row['world_refusals']),('ab'*32,TREE,8,8,[]))
            self.assertEqual((row['batch_size'],row['bucket'],row['block_causes']),(3,4,[None]*4))
            self.assertEqual(len(row['q_values']),9);self.assertEqual([q is None for q in row['q_values']],[not x for x in legal])
            self.assertTrue(all(-1<q<1 for q in row['q_values']if q is not None));self.assertTrue(legal[row['action_index']])
            acts=[a for a in range(9)if legal[a]]
            self.assertEqual(row['action_index'],max(acts,key=lambda a:(row['q_values'][a],-a)))
            self.assertEqual(row['action_policy_target'],max(acts,key=lambda a:(row['policy_target'][a],-a)))
            self.assertEqual(set(row['timings']),set(live.STAGES));self.assertTrue(all(type(v)is float and v>=0 for v in row['timings'].values()))
            self.assertTrue(type(row['decision_seconds'])is float and row['decision_seconds']>=0)
            self.assertEqual({k:row['pricer'][k]for k in('depth','k','k1','selection','W','K','buckets','capacities','fixed_capacity','precision','evaluator',
                'root_operator','root_operator_impl','matmul_precision')},dict(depth=2,k=[9,9],k1=[9,9],selection='prior',W=8,K=4,buckets=[1,2,4],
                capacities=[4,8,16],fixed_capacity=True,precision='bfloat16',evaluator=['champion','ab'*32,'bfloat16'],root_operator='ARGMAX_MEAN_COMPLETED_Q',
                root_operator_impl='search_hedge.averaging',matmul_precision='default'))
        # The same roots through constructor_worlds directly, on the same bucket's pricer: the played action and q_values are the
        # deployment cell's averaging over each root's 2K backed-up worlds (block order), the reported fields constructor_worlds.reduce's.
        constructor=self.cw.constructor(8)
        with Codex(outcomes):prepared=cw.prepare_roots(constructor,[{k:r[k]for k in live.PROTOCOL_KEYS}for r in reqs])
        made=cw.build(constructor,prepared,s.pricers[4],PARAMS,binding=self.cw.binding,recipe=self.cw.recipe,only_complete=True)
        frame,_=cw.price(s.pricers[4],made,self.cw.binding,self.cw.recipe,PARAMS)
        for row,direct,root in zip(rows,cw.reduce(frame,made),made.roots):
            self.assertEqual((row['completed_q'],row['policy_target'],row['root_value'],row['action_mean_q'],row['action_policy_target']),
                (list(direct['completed_q']),list(direct['policy_target']),direct['root_value'],direct['action_mean_q'],direct['action_policy_target']))
            prices=tuple(w for r in root['rows']for w in backups.backup(frame,r,backups.learner_max));choice=hedge.averaging(prices,root['root_legal'])
            self.assertEqual((row['action_index'],row['q_values']),(hedge.top_action(choice.strategy),list(choice.q)))

    def test_the_played_action_is_averagings_never_the_reported_moves(self):
        from gpu_engine import d3_search_latency as latency
        compose=latency.compose;legal=[a for a in range(9)if CW.T.ROOT_LEGAL[a]]
        def other(improvements,logits,root_legal,original):   # the composed completed-Q made to favour another action than averaging's
            out=compose(improvements,logits,root_legal,original);best=max(legal,key=lambda a:(out['completed_q'][a],-a));other=next(a for a in legal if a!=best)
            return dict(out,completed_q=tuple(None if q is None else(.9 if a==other else -.9)for a,q in enumerate(out['completed_q'])))
        with patch.object(latency,'compose',side_effect=other):rows,_=self.run_batch(self.search(),[request('t',0)],{('t',0):self.outcome(0)})
        row=rows[0];self.assertEqual(row['status'],'COMPLETE');self.assertFalse(row['mean_q_agrees']);self.assertNotEqual(row['action_index'],row['action_mean_q'])
        self.assertEqual(row['action_index'],max(legal,key=lambda a:(row['q_values'][a],-a)))   # averaging's argmax plays

    # root classes, refused classes, carry structure -----------------------------------------------------------------------------
    def test_root_classes_come_from_batch_2s_protocol_facts(self):
        fresh=dict(pokemon=[dict(species='Snorlax',condition='100/100',active=True)],volatiles=['substitute'],substituteHealthState='full',
            substituteDepletion=0,substituteMinDepletion=0,substituteOriginSpecies='Snorlax')
        sub_lines=('|switch|p2a: Snorlax|Snorlax, L80|100/100','|-start|p2a: Snorlax|Substitute')
        cases=dict(   # battle: (request keywords, public payload, classes, root_class, source)
            plain=(dict(),None,[],'other','protocol_facts'),
            forced=(dict(last=dict(forceSwitch=[True])),None,['forced_switch'],'forced_switch','protocol_facts'),   # the request; its replacement check TRIGGERED
            wish=(dict(),dict(payload(),wishSetTurns={'p2':1}),['wish'],'wish','protocol_facts'),
            sub=(dict(before=sub_lines),payload(p2=fresh),['fresh_substitute'],'fresh_substitute','protocol_facts'),
            stale=(dict(before=sub_lines),payload(p2=dict(fresh,substituteHealthState='damaged')),['fresh_substitute'],'fresh_substitute','protocol_facts'),
            asleep=(dict(),payload(p2=dict(pokemon=[dict(species='Snorlax',condition='100/100 slp',active=True)],volatiles=[])),['sleep'],'sleep','protocol_facts'),
            slept=(dict(before=('|-status|p2a: Snorlax|slp',)),None,['sleep'],'sleep','protocol_facts'),   # the public history (_sleep_origins)
            both=(dict(last=dict(forceSwitch=[True]),before=('|-status|p2a: Snorlax|slp',)),dict(payload(),wishSetTurns={'p1':0}),
                ['forced_switch','sleep','wish'],'forced_switch','protocol_facts'),
            forced_sub=(dict(last=dict(forceSwitch=[True]),before=sub_lines),payload(p2=fresh),['forced_switch','fresh_substitute'],'forced_switch','protocol_facts'),
            refused=(dict(last=dict(forceSwitch=[True])),None,['forced_switch'],'forced_switch','request_and_lines'))   # capture fails: request and lines only
        outcomes={(b,0):(Refusal('PUBLIC_REPLACEMENT_PHASE_UNAVAILABLE')if b=='refused'else self.outcome(i))for i,b in enumerate(cases)}
        payloads={(b,0):c[1]for b,c in cases.items()if c[1]is not None}
        with Codex(outcomes,payloads):rows=self.search().process_many([request(b,0,**c[0])for b,c in cases.items()])
        for(b,(_,_,classes,first,source)),row in zip(cases.items(),rows):
            with self.subTest(b):
                self.assertEqual((row['root_classes'],row['root_class'],row['root_class_source']),(classes,first,source))
                self.assertTrue(type(row['timings']['classify_seconds'])is float)
                self.assertEqual(row['status'],'REFUSED'if b=='refused'else'COMPLETE')   # classification decides nothing by itself
        with patch.object(live,'classify',side_effect=RuntimeError('boom')):rows,_=self.run_batch(self.search(),[request('plain',0)],{('plain',0):self.outcome(0)})
        self.assertEqual((rows[0]['status'],rows[0]['root_classes'],rows[0]['root_class'],rows[0]['root_class_source']),('COMPLETE',None,None,'UNCLASSIFIED: RuntimeError: boom'))
        self.assertEqual([live.root_class(c)for c in([],['sleep','wish'],['fresh_substitute','sleep','wish'])],['other','wish','fresh_substitute'])

    def test_refused_classes_are_refused_before_pricing(self):
        s=self.search(refuse_classes=('sleep','forced_switch'))
        self.assertEqual(s.refuse_classes,('forced_switch','sleep'))
        outcomes={('f',0):self.outcome(0),('p',0):self.outcome(1),('r',0):Refusal('PUBLIC_REPLACEMENT_PHASE_UNAVAILABLE'),('fs',0):self.outcome(2)}
        rows,_=self.run_batch(s,[request('f',0,last=dict(forceSwitch=[True])),request('p',0),request('r',0,last=dict(forceSwitch=[True])),
            request('fs',0,last=dict(forceSwitch=[True]),before=('|-status|p2a: Snorlax|slp',))],outcomes)
        self.assertEqual([(r['status'],r['cause'],r['refusal'])for r in rows],[('REFUSED','class_refused','ROOT_CLASS_REFUSED:forced_switch'),('COMPLETE',None,None),
            ('REFUSED','constructor_refused','constructor-root:PUBLIC_REPLACEMENT_PHASE_UNAVAILABLE'),('REFUSED','class_refused','ROOT_CLASS_REFUSED:forced_switch,sleep')])
        self.assertEqual((rows[0]['constructed_worlds'],rows[1]['bucket'],s.pricers[1].last_timing['roots']),(8,1,4))   # one root priced, no device work for the others
        self.assertIn('class_refused',live.CAUSES)
        with self.assertRaisesRegex(ValueError,'refuse_classes'):self.search(refuse_classes=('nap',))
        # A root with a refused world keeps the constructor's cause: no class gate, no structure check (nor a structural count).
        outcomes={('part',0):self.outcome(0,refused=(3,)),('oddpart',0):self.outcome(1,carry='plain',refused=(3,)),('ok',0):self.outcome(2)}
        rows,_=self.run_batch(self.search(refuse_classes=('forced_switch',)),[request('part',0,last=dict(forceSwitch=[True])),request('ok',0),
            request('oddpart',0)],outcomes)
        self.assertEqual([(r['status'],r['cause'],r['refusal'])for r in rows],[('REFUSED','constructor_refused','WORLD_CONSTRUCTOR_REFUSED:named-refusal'),
            ('COMPLETE',None,None),('REFUSED','constructor_refused','WORLD_CONSTRUCTOR_REFUSED:named-refusal')])
        self.assertEqual([r['timings']['structure_seconds']is None for r in rows],[True,False,True])
        # Classes refused and a root that could not be classified: ROOT_CLASS_UNKNOWN; priced when no class is refused.
        with patch.object(live,'classify',side_effect=RuntimeError('boom')):
            rows,_=self.run_batch(self.search(refuse_classes=('wish',)),[request('u',0)],{('u',0):self.outcome(0)})
        self.assertEqual((rows[0]['cause'],rows[0]['refusal'],rows[0]['root_classes']),('class_refused','ROOT_CLASS_UNKNOWN',None))

    def test_the_recomputed_classes_are_cross_checked_against_the_worlds_restore_checks(self):
        def diagnosed(i,*kinds,sleep=False):
            d,m,l=self.outcome(i);checks=[dict(kind=k,status='CLEAR',fatal=True)for k in kinds]
            checks.append(dict(kind='PUBLIC_SLEEP_WORLD_POSTERIOR',status='CLEAR',fatal=True,applicability='CURRENT_SLEEP'if sleep else'NO_CURRENT_SLEEP'))
            return d,replace(m,mapper_diagnostics=tuple(dict(world=w,checks=list(checks))for w in range(len(m.worlds)))),l
        outcomes={('f',0):diagnosed(0,'PUBLIC_REPLACEMENT_PHASE'),('s',0):diagnosed(1,sleep=True),('w',0):diagnosed(2,'PUBLIC_WISH_RECIPIENT_STATE'),
            ('n',0):self.outcome(3)}
        rows,_=self.run_batch(self.search(),[request('f',0,last=dict(forceSwitch=[True])),request('s',0,before=('|-status|p2a: Snorlax|slp',)),
            request('w',0),request('n',0)],outcomes)
        self.assertEqual([r['root_class_crosscheck']for r in rows],['agrees','agrees',"DISAGREES: recomputed only []; worlds only ['wish']",None])
        self.assertEqual({r['status']for r in rows},{'COMPLETE'})   # reported only: the recomputed classes stay the source
        self.assertEqual(rows[2]['root_classes'],[])

    def test_a_root_whose_carry_differs_is_refused_alone(self):
        outcomes={('ok',0):self.outcome(0),('odd',0):self.outcome(1,carry='plain'),('w0',0):self.outcome(2)}   # odd: the fake's own carry, not the constructor's
        s=self.search();rows,_=self.run_batch(s,[request('ok',0),request('odd',0)],outcomes)
        self.assertEqual([(r['status'],r['cause'],r['bucket'])for r in rows],[('COMPLETE',None,1),('REFUSED','depthn_refused',1)])
        self.assertTrue(rows[1]['refusal'].startswith("WORLD_STRUCTURE_MISMATCH:world 0: missing [1].features['choice_locked']"),rows[1]['refusal'])
        self.assertTrue(s.carry['provisional'])   # no warmup: the first constructed world seen
        s=self.search();self.run_batch(s,[request('ok',0)],outcomes);self.assertTrue(s.carry['provisional'])   # a provisional reference first
        with Codex(outcomes):report=s.warmup([request('w0',0)])
        self.assertEqual((s.carry['provisional'],report['carry_signature']),(False,s.carry['signature']))   # warmup replaced it with its own root
        self.assertEqual(report['carry_reference'],dict(battle_id='w0',owner='p1',stream_id=7,decision_index=0,prefix_sha256=request('w0',0)['prefix_sha256'],
            root_classes=[]))
        rows,_=self.run_batch(s,[request('odd',0),request('ok',0)],outcomes)   # warmed: the recorded reference, whichever root comes first
        self.assertEqual([(r['status'],r['cause'])for r in rows],[('REFUSED','depthn_refused'),('COMPLETE',None)])
        # Without the check the two would share one frame, which DepthN refuses as a whole.
        constructor=self.cw.constructor(8)
        with Codex(outcomes):prepared=cw.prepare_roots(constructor,[{k:r[k]for k in live.PROTOCOL_KEYS}for r in(request('ok',0),request('odd',0))])
        made=cw.build(constructor,prepared,s.pricers[2],PARAMS,binding=self.cw.binding,recipe=self.cw.recipe,only_complete=True)
        with self.assertRaisesRegex(ValueError,'one world source per frame'):cw.price(s.pricers[2],made,self.cw.binding,self.cw.recipe,PARAMS)

    def test_warmup_holds_every_root_to_one_carry_and_its_programs_to_it(self):
        outcomes={('odd',0):self.outcome(1,carry='plain'),('ok',0):self.outcome(0),('ok2',0):self.outcome(2)}
        with Codex(outcomes),self.assertRaisesRegex(RuntimeError,r"every warmup root must share one carry structure; against the reference root \{'battle_id': 'odd'"):
            self.search().warmup([request('odd',0),request('ok',0)])   # the first root seen is never adopted silently
        # At bucket 1 only (the bench's setup) the priced window is the first root alone: every other warmup root is still checked.
        single=self.make_pricers(8,(1,));s=self.search(pricers=single)
        with Codex(outcomes),self.assertRaisesRegex(RuntimeError,r"against the reference root \{'battle_id': 'odd'.*ok/0: world 0: extra \[1\].features"):
            s.warmup([request('odd',0),request('ok',0)])
        self.assertEqual((s.constructor.seen,s.constructor.sessions,s.battles),(set(),{},{}))   # sessions and attempts reset after it
        with Codex(outcomes):report=self.search(pricers=self.make_pricers(8,(1,))).warmup([request('ok',0),request('ok2',0)])   # one structure: fine
        self.assertEqual(report['carry_reference']['battle_id'],'ok')
        # A program table already holding another carry's programs is not this warmup's: the signature check refuses it.
        pricers=self.make_pricers(8,(1,));self.run_batch(self.search(pricers=pricers),[request('odd',0)],outcomes)   # compiles the odd carry's programs
        with Codex(outcomes),self.assertRaisesRegex(RuntimeError,"is not the warmed programs"):self.search(pricers=pricers).warmup([request('ok',0)])

    def test_a_structural_exception_at_bucket_1_is_that_roots_server_error(self):
        error=ValueError('one world source per frame: pre-built worlds with different (state, public) structures')
        with patch.object(live.cw,'price',side_effect=error):
            rows,_=self.run_batch(self.search(),[request('ok',0,last=dict(forceSwitch=[True]))],{('ok',0):self.outcome(0)})
        self.assertEqual((rows[0]['cause'],rows[0]['refusal'],rows[0]['bucket'],rows[0]['root_classes'],rows[0]['root_class']),
            ('server_error',f'ValueError: {error}',None,['forced_switch'],'forced_switch'))   # still classified: the bench counts it per class
        self.assertTrue(type(rows[0]['timings']['classify_seconds'])is float);encode(rows[0])

    def test_bucket_is_the_smallest_holding_the_priced_blocks_and_long_lists_chunk(self):
        for n,bucket in((1,1),(2,2),(3,4),(4,4)):
            s=self.search();outcomes={(f'n{i}',0):self.outcome(i)for i in range(n)}
            rows,_=self.run_batch(s,[request(f'n{i}',0)for i in range(n)],outcomes)
            self.assertEqual({(r['status'],r['bucket'],r['batch_size'])for r in rows},{('COMPLETE',bucket,n)})
            self.assertEqual((s.pricers[bucket].last_timing['root_capacity'],s.pricers[bucket].last_timing['roots']),(4*bucket,4*n))
        s=self.search();outcomes={('a',0):self.outcome(0),('b',0):Refusal('SEARCH_PREFIX_OR_REQUEST_MISMATCH'),('c',0):self.outcome(2)}
        rows,_=self.run_batch(s,[request(x,0)for x in'abc'],outcomes)   # one constructor refusal: two roots priced, bucket 2
        self.assertEqual([(r['status'],r['bucket'])for r in rows],[('COMPLETE',2),('REFUSED',2),('COMPLETE',2)])
        s=self.search(max_requests=2);outcomes={(f'c{i}',0):self.outcome(i)for i in range(5)}
        rows,_=self.run_batch(s,[request(f'c{i}',0)for i in range(5)],outcomes)
        self.assertEqual([(r['batch_size'],r['bucket'])for r in rows],[(2,2)]*4+[(1,1)]);self.assertEqual({r['status']for r in rows},{'COMPLETE'})
        s=self.search(16);rows,_=self.run_batch(s,[request('w',0,rung='DN2W16')],{('w',0):self.outcome(0,16)})
        self.assertEqual((rows[0]['status'],rows[0]['rung'],rows[0]['requested_worlds'],rows[0]['constructed_worlds'],rows[0]['bucket'],rows[0]['pricer']['K']),
            ('COMPLETE','DN2W16',16,16,1,8));self.assertEqual(rows[0]['block_causes'],[None]*8)

    # refusals ------------------------------------------------------------------------------------------------------------------
    def test_bad_requests_are_refused_unprepared(self):
        good=request('b0',0);drop=lambda k:{x:v for x,v in good.items()if x!=k}
        cases=dict(not_a_dict=['b0'],no_rung=drop('rung'),no_budget=drop('budget_seconds'),no_lines=drop('lines'),extra=dict(good,extra=1),
            schema=dict(good,schema='other'),request_id=dict(good,request_id=5),empty_id=dict(good,request_id=''),rung_type=dict(good,rung=8),
            budget_zero=dict(good,budget_seconds=0.),budget_negative=dict(good,budget_seconds=-1),budget_nan=dict(good,budget_seconds=float('nan')),
            budget_inf=dict(good,budget_seconds=float('inf')),budget_bool=dict(good,budget_seconds=True),budget_str=dict(good,budget_seconds='3'),
            arrival=dict(good,**{live.ARRIVAL:'soon'}))
        s=self.search();rows,codex=self.run_batch(s,list(cases.values()),{})
        for name,row in zip(cases,rows):
            with self.subTest(name):
                self.assertEqual((row['status'],row['cause'],set(row)),('REFUSED','bad_request',KEYS));self.assertTrue(row['refusal'].startswith('MALFORMED_REQUEST'))
                self.assertEqual(row['request_id'],cases[name]['request_id']if isinstance(cases[name],dict)and type(cases[name].get('request_id'))is str else None)
                encode(row)
        self.assertEqual((codex.captures,s.constructor.seen,s.battles),([],set(),{}))

    def test_a_rung_this_server_does_not_serve_is_a_bad_request(self):
        s=self.search();outcomes={('b0',0):self.outcome(0)}
        rows,codex=self.run_batch(s,[request('b0',0,rung='DN2W16')],outcomes)
        self.assertEqual((rows[0]['cause'],rows[0]['rung']),('bad_request','DN2W16'));self.assertTrue(rows[0]['refusal'].startswith("RUNG_NOT_SERVED: 'DN2W16'"))   # the client's rung, echoed
        rows,_=self.run_batch(s,[dict(request('b0',0),rung=8)],outcomes);self.assertEqual((rows[0]['cause'],rows[0]['rung']),('bad_request','DN2W8'))   # not a string: ours
        self.assertEqual((codex.captures,s.constructor.seen),([],set()))
        rows,_=self.run_batch(s,[request('b0',0)],outcomes);self.assertEqual(rows[0]['status'],'COMPLETE')   # never attempted, so not ALREADY_ATTEMPTED

    def test_a_request_whose_queue_wait_leaves_too_little_budget_expires_unpriced(self):
        s=self.search(clock=lambda:1000.,min_price_seconds=1.);outcomes={(f'e{i}',0):self.outcome(i)for i in range(5)}
        reqs=[dict(request('e0',0),**{live.ARRIVAL:997.5}),dict(request('e1',0),**{live.ARRIVAL:998.5}),request('e2',0),
            dict(request('e3',0,budget=1.),**{live.ARRIVAL:1000.}),request('e4',0,budget=.5)]
        rows,codex=self.run_batch(s,reqs,outcomes)
        self.assertEqual([r['cause']for r in rows],['expired_in_queue',None,None,None,'expired_in_queue'])
        self.assertEqual([r['timings']['queue_seconds']for r in rows],[2.5,1.5,0.,0.,0.]);self.assertTrue(rows[0]['refusal'].startswith('EXPIRED_IN_QUEUE'))
        self.assertEqual(sorted(set(codex.captures)),[('e1',0),('e2',0),('e3',0)]);self.assertEqual(rows[1]['bucket'],4)
        rows,_=self.run_batch(s,[dict(request('e0',0),**{live.ARRIVAL:999.9})],outcomes);self.assertEqual(rows[0]['status'],'COMPLETE')   # never attempted

    def test_constructor_refusals(self):
        s=self.search()
        outcomes={('p',0):Refusal('SEARCH_PREFIX_OR_REQUEST_MISMATCH'),('w',0):self.outcome(1,refused=(3,)),('ok',0):self.outcome(2),('m',0):self.outcome(3)}
        rows,_=self.run_batch(s,[request('p',0),request('w',0),request('ok',0),request('m',0,stream=0)],outcomes)   # stream 0: Codex's own address check
        self.assertEqual([(r['status'],r['cause'])for r in rows],[('REFUSED','constructor_refused'),('REFUSED','constructor_refused'),('COMPLETE',None),
            ('REFUSED','constructor_refused')])
        self.assertEqual([r['refusal']for r in rows],['constructor-root:SEARCH_PREFIX_OR_REQUEST_MISMATCH','WORLD_CONSTRUCTOR_REFUSED:named-refusal',None,
            'constructor-root:MALFORMED_SEARCH_ADDRESS'])
        self.assertEqual((rows[1]['constructed_worlds'],rows[1]['world_refusals'],rows[1]['bucket']),(7,[dict(world=3,kinds=['named-refusal'])],1))
        self.assertEqual((rows[0]['constructed_worlds'],rows[0]['world_refusals']),(0,[]))
        rows,_=self.run_batch(s,[dict(request('ok',0),request_id='again')],outcomes)   # Codex's no-retry rule holds
        self.assertEqual((rows[0]['cause'],rows[0]['refusal'],rows[0]['bucket']),('constructor_refused','constructor-root:SEARCH_REQUEST_ALREADY_ATTEMPTED',None))

    def test_complete_responses_carry_the_effect_guard_record(self):
        """The pricers' observer seam feeds the record of every COMPLETE response (the fake engine's labels; surf guarded for the
        test): the peer's Surf trips it at turn 1 and turn 2; a peer without it does not; a refused root has none."""
        from pokezero import depthn_effect_guard as G
        from tests.test_gpu_engine_depthn_guard_inputs import fake_labels, TACKLE, REST, PROTECT
        make=lambda labels:live.make_pricers(self.executor,self.evaluator,PARAMS,worlds=8,buckets=(1,),block_lanes=81,blocks_per_call=4,labels=labels)
        outcomes={('g',0):self.outcome(0),('r',0):Refusal('SEARCH_PREFIX_OR_REQUEST_MISMATCH')}
        with patch.dict(G.GUARDED,surf=('A','test',None)),patch.object(G,'WATCHED',G.WATCHED|{'surf'}):
            rows,_=self.run_batch(self.search(pricers=make(fake_labels)),[request('g',0),request('r',0)],outcomes)
            clear,_=self.run_batch(self.search(pricers=make(lambda s:fake_labels(s,peer_moves=(TACKLE,REST,PROTECT,0)))),[request('g',0)],outcomes)
        guard=rows[0]['effect_guard'];encode(rows[0])
        self.assertEqual((rows[0]['status'],guard['tripped'],guard['b_turn1'],guard['b_turn2'],guard['choice']),('COMPLETE',True,True,True,rows[0]['action_index']))
        self.assertEqual(guard['turn2_cells']['p1_captures'],1)
        # reasons list what tripped (turn 1 and the reach, its tier A mass from Surf); the any-cell hits sit under any_cell_reasons
        self.assertEqual({(x['move'],x['side'],x['source'])for x in guard['reasons']},{('surf','opp','turn1'),(None,'opp','reach')})
        reach=next(x for x in guard['reasons'] if x['source']=='reach')
        self.assertEqual((reach['tiers'],list(reach['moves']),guard['tiers']),({'A':guard['m_opp']},['surf'],['A']))
        self.assertEqual({(x['move'],x['side'],x['source'])for x in guard['any_cell_reasons']},{('surf','opp','turn2')})
        # the live rule (reach): M_opp over the frame's own priors trips the turn-2 arm; the any-cell arm is kept for analysis
        self.assertEqual((guard['rule'],guard['arms'],guard['m_own'],guard['any_cell_b'],guard['any_cell_tripped']),('reach',['turn1','reach'],0.,True,True))
        self.assertGreaterEqual(guard['m_opp'],G.THRESHOLD);self.assertEqual(guard['reach']['m_opp'],guard['m_opp'])
        self.assertEqual((rows[1]['status'],rows[1]['effect_guard']),('REFUSED',None))
        self.assertEqual((clear[0]['status'],clear[0]['effect_guard']['tripped'],clear[0]['effect_guard']['reasons']),('COMPLETE',False,[]))
        self.assertEqual((clear[0]['effect_guard']['m_own'],clear[0]['effect_guard']['m_opp'],clear[0]['effect_guard']['arms']),(0.,0.,[]))

    def test_a_root_whose_opponent_must_recharge_is_constructor_refused(self):
        """The lead's decision: the peer's -mustrecharge on the owner's lines refuses the root in prepare_roots (cause
        constructor_refused, refusal constructor-root:OPPONENT_RECHARGE_UNREPRESENTED:...); the owner's own does not."""
        s=self.search();beam=lambda seat:(f'|move|{seat}a: Snorlax|Hyper Beam|{"p2"if seat=="p1"else"p1"}a: Snorlax',f'|-mustrecharge|{seat}a: Snorlax')
        outcomes={('peer',0):self.outcome(0),('own',0):self.outcome(1)}
        rows,_=self.run_batch(s,[request('peer',0,before=beam('p2')),request('own',0,before=beam('p1'))],outcomes)
        self.assertEqual([(r['status'],r['cause'])for r in rows],[('REFUSED','constructor_refused'),('COMPLETE',None)])
        self.assertEqual(rows[0]['refusal'],'constructor-root:OPPONENT_RECHARGE_UNREPRESENTED:'+cw.OPPONENT_RECHARGE_DETAIL)
        self.assertEqual((rows[0]['constructed_worlds'],rows[0]['world_refusals']),(0,[]))

    def test_depthn_refusals(self):
        s=self.search();empty=self.outcome(1);empty=empty[:2]+((False,)*9,)
        outcomes={('pre',0):self.outcome(0,values={5:.5}),('empty',0):empty,('ok',0):self.outcome(2),('bad',0):self.outcome(3)}
        rows,_=self.run_batch(s,[request('pre',0),request('empty',0)],outcomes)
        self.assertEqual([(r['cause'],r['refusal'],r['block_causes'])for r in rows],[('depthn_refused','pre',[None,None,'pre',None]),
            ('depthn_refused','ROOT_LEGAL_EMPTY',None)])
        # The reporting reduction failing (reduce_completed_q's range check) leaves the decision: the root still plays averaging's action.
        from gpu_engine import d3_search_latency as latency
        reduce_block=latency.reduce_block;calls=[]
        def failing(prices,logits,legal,original):
            calls.append(1)
            if len(calls)>4:raise ValueError('finite in-range numerical value required')   # the second root's blocks
            return reduce_block(prices,logits,legal,original)
        with patch.object(latency,'reduce_block',side_effect=failing):rows,_=self.run_batch(self.search(),[request('ok',0),request('bad',0)],outcomes)
        self.assertEqual([(r['status'],r['cause'],r['reduction_note'])for r in rows],[('COMPLETE',None,None),
            ('COMPLETE',None,'REDUCTION_ERROR: ValueError: finite in-range numerical value required')])
        self.assertEqual([rows[1][k]for k in('completed_q','policy_target','root_value','action_mean_q','action_policy_target','mean_q_agrees')],[None]*6)
        self.assertTrue(rows[1]['q_values']and rows[1]['action_index']is not None)
        # Averaging's own failures refuse the root: its error, an unknown price, a non-finite result, an invalid world price.
        averaging=hedge.averaging
        cases=dict(error=(live.hedge,'averaging',lambda p,l:(_ for _ in()).throw(ValueError('empty mixed strategy')),'AVERAGING_ERROR: empty mixed strategy'),
            unknown=(live.hedge,'averaging',lambda p,l:None,'AVERAGING_UNKNOWN_PRICE'),
            nan=(live.hedge,'averaging',lambda p,l:(lambda c:hedge.Choice(c.strategy,c.value,tuple(None if q is None else math.nan for q in c.q)))(averaging(p,l)),
                'AVERAGING_INVALID_RESULT'),
            value=(live.backups,'backup',lambda frame,row,rule:(_ for _ in()).throw(ValueError('public leaf value outside[-1,1]')),
                'AVERAGING_ERROR: public leaf value outside[-1,1]'))
        for name,(target,attr,fake,reason)in cases.items():
            with self.subTest(name),patch.object(target,attr,side_effect=fake):rows,_=self.run_batch(self.search(),[request('ok',0)],outcomes)
            self.assertEqual((rows[0]['status'],rows[0]['cause'],rows[0]['refusal'],rows[0]['action_index']),('REFUSED','depthn_refused',reason,None))
        # Any failure of the reporting reduction only blanks the report; MemoryError still propagates.
        for error in(RuntimeError('report broke'),TypeError('report broke'),KeyError('report broke')):
            with self.subTest(type(error).__name__),patch.object(latency,'compose',side_effect=error):rows,_=self.run_batch(self.search(),[request('ok',0)],outcomes)
            self.assertEqual((rows[0]['status'],rows[0]['reduction_note'],rows[0]['action_mean_q']),('COMPLETE',f'REDUCTION_ERROR: {type(error).__name__}: {error}',None))
        with patch.object(latency,'compose',side_effect=MemoryError()),self.assertRaises(MemoryError):self.run_batch(self.search(),[request('ok',0)],outcomes)
        with self.assertRaisesRegex(ValueError,'outside'):backups.original.WorldPrices(False,(False,)*9,(0.,)*9,((1.5,)+(None,)*8,)+((None,)*9,)*8)
        with self.assertRaisesRegex(ValueError,'finite'):backups.original.WorldPrices(False,(False,)*9,(0.,)*9,((math.nan,)+(None,)*8,)+((None,)*9,)*8)

    def test_a_batch_exception_refuses_every_request_of_the_batch_server_error(self):
        outcomes={(f'x{i}',0):self.outcome(i)for i in range(2)};bad=dict(request('y',0),schema='other')
        old=dict(request('z',0),**{live.ARRIVAL:0.})   # expired: waited since monotonic 0
        for target,error in(('price',RuntimeError('device lost')),('prepare_roots',TypeError('prepare changed')),('build',ValueError('binding differs'))):
            s=self.search()
            with patch.object(live.cw,target,side_effect=error):rows,_=self.run_batch(s,[request('x0',0),bad,request('x1',0),old],outcomes)
            self.assertEqual([r['cause']for r in rows],['server_error','bad_request','server_error','expired_in_queue'])
            self.assertEqual(rows[0]['refusal'],f'{type(error).__name__}: {error}');self.assertIsNotNone(rows[0]['timings']['queue_seconds'])
            self.assertEqual(set(s.battles),{'x0','x1'})   # still accounted for eviction
        s=self.search()
        with patch.object(live.cw,'price',side_effect=MemoryError()),self.assertRaises(MemoryError):self.run_batch(s,[request('x0',0)],outcomes)
        from gpu_engine import d3_search_latency as latency
        compose=latency.compose
        with patch.object(latency,'compose',side_effect=lambda *a:dict(compose(*a),root_value=math.nan)):rows,_=self.run_batch(self.search(),[request('x0',0)],outcomes)
        self.assertEqual(rows[0]['cause'],'server_error');self.assertTrue(rows[0]['refusal'].startswith('RESPONSE_NOT_ENCODABLE'));encode(rows[0])

    # sessions ------------------------------------------------------------------------------------------------------------------
    def test_sessions_evict_the_least_recent_battles_and_keep_live_ones(self):
        s=self.search(max_requests=2,session_cap=2)
        outcomes={(b,d):self.outcome(i)for i,(b,d)in enumerate((('A',0),('B',0),('C',0),('B',1),('A',1),('D',0),('E',0)))}
        battles=lambda:({k[1]for k in s.constructor.sessions},{a[0][1]for a in s.constructor.seen})
        with Codex(outcomes)as codex:
            for b in'ABC':self.assertEqual(s.process_many([request(b,0)])[0]['status'],'COMPLETE')
            self.assertEqual(battles(),({'B','C'},{'B','C'}));self.assertEqual(s.evicted,1)
            self.assertTrue(all(type(k)is tuple and k[0]=='DN2W8'for k in s.constructor.sessions))   # Codex's (rung, battle, owner, stream) keys
            adapters=len(codex.adapters);b_adapter=codex.adapters[1]
            self.assertEqual(s.process_many([request('B',1)])[0]['status'],'COMPLETE')   # B's fold continues incrementally
            self.assertEqual(len(codex.adapters),adapters);self.assertEqual(b_adapter.fed,[x for x in prefix('B',1)if not x.startswith('|request|')])
            row=s.process_many([dict(request('B',0),request_id='B:retry')])[0]   # a live battle keeps its no-retry rule
            self.assertEqual(row['refusal'],'constructor-root:SEARCH_REQUEST_ALREADY_ATTEMPTED')
            self.assertEqual(s.process_many([request('A',1)])[0]['status'],'COMPLETE')   # evicted A returns: a fresh adapter folds its full prefix
            fresh=codex.adapters[-1];self.assertEqual((fresh.resets,fresh.fed),([(7,'A')],[x for x in prefix('A',1)if not x.startswith('|request|')]))
            self.assertEqual(battles(),({'A','B'},{'A','B'}))
            rows=s.process_many([request('D',0),request('E',0)])   # the current batch's battles are never the ones evicted
        self.assertEqual([r['status']for r in rows],['COMPLETE']*2);self.assertEqual(battles(),({'D','E'},{'D','E'}));self.assertEqual(list(s.battles),['D','E'])
        with self.assertRaisesRegex(ValueError,'session cap'):self.search(max_requests=4,session_cap=2)

    # warmup --------------------------------------------------------------------------------------------------------------------
    def test_warmup_compiles_every_bucket_before_live_requests(self):
        pricers=self.make_pricers(8,(1,2,4));s=self.search(pricers=pricers)
        outcomes={('w0',0):self.outcome(0),('w1',0):self.outcome(1),**{(f'l{i}',0):self.outcome(2+i)for i in range(3)}}
        with Codex(outcomes):report=s.warmup([request('w0',0),request('w1',0)])   # two saved roots: bucket 4 cycles them (repeats refused, still priced at 4)
        self.assertEqual(set(report['buckets']),{1,2,4});json.dumps(report)
        for b,entry in report['buckets'].items():
            self.assertGreater(entry['cold_seconds'],0);self.assertGreater(entry['warm_seconds'],0);self.assertEqual(entry['warm_compiled'],0)
            self.assertEqual(entry['complete'],min(b,2))
        self.assertGreater(report['buckets'][1]['compiled'],0);self.assertGreater(report['buckets'][4]['compiled'],0)
        self.assertEqual(report['buckets'][4]['causes'],['constructor_refused'])
        self.assertEqual(report['programs'],len(pricers[1].programs));self.assertEqual({k[1]for k in pricers[1].programs if k[0]=='P1'},{4,8,16})
        self.assertEqual((s.constructor.seen,s.constructor.sessions,s.battles),(set(),{},{}));self.assertIs(s.warm,report)
        self.assertEqual([s.missing_programs(b)for b in(1,2,4)],[[]]*3)   # P1, selection, gather and the leaf program, per bucket
        # the effect guard's label gather compiled at both shapes: each bucket's P1 children and the root worlds [W, 1, ...]
        self.assertEqual(report['labels'],dict(p1_rows={1:2*4*81,2:2*8*81,4:2*16*81},root_rows=8))
        self.assertTrue(all(2*b*4*81 in pricers[b].label_rows for b in(1,2,4)))
        for n,b in((1,1),(2,2),(3,4)):
            rows,_=self.run_batch(s,[dict(request(f'l{i}',0),request_id=f'l{i}:{n}')for i in range(n)],outcomes)
            self.assertEqual({(r['status'],r['bucket'])for r in rows},{('COMPLETE',b)});self.assertEqual(pricers[b].compiles,())
            s.reset_sessions()
        with Codex({('r',0):Refusal('WORLD_CONSTRUCTOR_REFUSED','all')}),self.assertRaisesRegex(RuntimeError,'no saved request priced at bucket 1'):
            self.search().warmup([request('r',0)])
        # A bucket priced without reaching the leaf program (here: its compiled leaf programs dropped) is not warm: warmup raises.
        bare=self.make_pricers(8,(1,));price=cw.price
        def leafless(pricer,*a,**kw):
            out=price(pricer,*a,**kw)
            for k in[k for k in pricer.programs if k[0]=='leaf']:del pricer.programs[k]
            return out
        with Codex(outcomes),patch.object(live.cw,'price',side_effect=leafless),self.assertRaisesRegex(RuntimeError,r'bucket 1 with every program compiled \(missing: leaf\)'):
            self.search(pricers=bare).warmup([request('w0',0)])
        # a label gather that fails on the pricer's children never compiled: warmup raises before READY
        broken=live.make_pricers(self.executor,self.evaluator,PARAMS,worlds=8,buckets=(1,),block_lanes=81,blocks_per_call=4,labels=lambda s:1/0)
        with Codex(outcomes),self.assertRaisesRegex(RuntimeError,r"label gather did not compile for buckets \[1\]"):
            self.search(pricers=broken).warmup([request('w0',0)])
        with self.assertRaisesRegex(ValueError,'saved requests with'):self.search().warmup([{'battle_id':'x'}])
        with self.assertRaisesRegex(ValueError,'at least one'):self.search().warmup([])

    def test_configuration_is_checked(self):
        bad=dict(depth=lambda W:{1:self.cw.pricer(self.executor,depth=3,k_own=9,k_opp=9,capacity=4,fixed_capacity=True,forward=self.evaluator)},
            width=lambda W:{1:self.cw.pricer(self.executor,depth=2,k_own=2,k_opp=2,capacity=4,fixed_capacity=True,forward=self.evaluator)},
            capacity=lambda W:{1:self.cw.pricer(self.executor,depth=2,k_own=9,k_opp=9,capacity=8,fixed_capacity=True,forward=self.evaluator)},
            fixed=lambda W:{1:self.cw.pricer(self.executor,depth=2,k_own=9,k_opp=9,capacity=4,forward=self.evaluator)},
            v2=lambda W:{1:self.cw.pricer(self.executor,depth=2,k_own=9,k_opp=9,capacity=4,fixed_capacity=True)})
        for name,make in bad.items():
            with self.subTest(name),self.assertRaises(ValueError):self.search(pricers=make(8))
        with self.assertRaisesRegex(ValueError,'W in'):self.search(2,pricers=self.pricers[8])
        with self.assertRaisesRegex(ValueError,'max_requests'):self.search(max_requests=5)
        with self.assertRaisesRegex(ValueError,'tree commit'):self.search(tree_commit='')
        with self.assertRaisesRegex(ValueError,'min_price_seconds'):self.search(min_price_seconds=float('nan'))
        other=self.cw.constructor(8,metadata=dict(self.cw.metadata,moves=[None,*[dict(pp=11)]*4]))
        with self.assertRaisesRegex(ValueError,'another binding'):
            live.DepthNLiveSearch(constructor=other,pricers=self.pricers[8],params=PARAMS,binding=self.cw.binding,recipe=self.cw.recipe,tree_commit=TREE)
        for bad in('ARGMAX_POLICY_TARGET','target_argmax',None):
            with self.subTest(root_operator=bad),self.assertRaisesRegex(ValueError,'root operator'):self.search(root_operator=bad)

    # the root operator (screen-r4b) ----------------------------------------------------------------------------------------------
    def operator_rows(self,operator,requests,outcomes,*,target=None):
        """Responses (stage timings 0) and the decide calls' (root_logits, root_legal) under `operator`; `target` forces the
        reported policy_target's argmax (favour)."""
        decided=[]
        def spy(blocks,root_logits,root_legal,**kw):decided.append((tuple(root_logits),tuple(root_legal),kw));return decide(blocks,root_logits,root_legal,**kw)
        decide=live.decide
        with ExitStack()as stack:
            still_clocks(stack);stack.enter_context(patch.object(live,'decide',side_effect=spy))
            if target is not None:stack.enter_context(favour(target))
            rows,_=self.run_batch(self.search(root_operator=operator,clock=lambda:0.),requests,outcomes)
        return rows,decided

    def test_the_default_operator_is_unchanged_and_target_argmax_changes_only_the_played_action(self):
        """The same roots under the default and TARGET_ARGMAX (its reported target forced to favour action 0, where averaging
        plays 2): the default responses carry the default operator and no target_fallback (KEYS, as before); a TARGET_ARGMAX
        response names its operator (and its pricer identity does) and adds target_fallback; its played action is the RR
        scorer's own target_argmax on the response's own policy_target and q_values (as the wire carries them) with the root's
        legal mask and logits; everything else is the default's, the effect guard record but its `choice` (now the played
        action) included."""
        from scripts import depthn_rr_score as scorer
        reqs=[request(f'b{i}',0,stream=10+i)for i in range(3)]+[request('r',0)]
        outcomes={**{(f'b{i}',0):self.outcome(i)for i in range(3)},('r',0):Refusal('SEARCH_PREFIX_OR_REQUEST_MISMATCH')}
        default,_=self.operator_rows(live.ROOT_OPERATOR,reqs,outcomes,target=0)
        plain,_=self.operator_rows(live.ROOT_OPERATOR,reqs,outcomes)   # under the default the forced target is reported only: averaging plays
        self.assertEqual([r['action_index']for r in plain],[2,2,2,None]);self.assertEqual([r['action_index']for r in default],[2,2,2,None])
        rows,decided=self.operator_rows(TARGET,reqs,outcomes,target=0)
        self.assertEqual([kw for *_,kw in decided],[dict(operator=TARGET)]*3)
        for i,(base,row)in enumerate(zip(default,rows)):
            with self.subTest(i):
                encode(base);encode(row);self.assertEqual(set(base),KEYS);self.assertEqual(set(row),KEYS|{'target_fallback'})
                self.assertEqual((base['root_operator'],base['root_operator_impl'],base['pricer']['root_operator'],base['pricer']['root_operator_impl']),
                    ('ARGMAX_MEAN_COMPLETED_Q','search_hedge.averaging','ARGMAX_MEAN_COMPLETED_Q','search_hedge.averaging'))
                self.assertEqual((row['root_operator'],row['root_operator_impl'],row['pricer']['root_operator'],row['pricer']['root_operator_impl']),
                    (TARGET,'depthn_root_operator.target_argmax',TARGET,'depthn_root_operator.target_argmax'))
                self.assertEqual({k:v for k,v in row['pricer'].items()if not k.startswith('root_operator')},
                    {k:v for k,v in base['pricer'].items()if not k.startswith('root_operator')})
                same=lambda r:{k:v for k,v in r.items()if k not in(*OPERATOR_KEYS,'pricer','action_index','effect_guard')}
                self.assertEqual(same(row),same(base))
                if base['status']!='COMPLETE':
                    self.assertEqual((row['status'],row['action_index'],row['target_fallback'],row['effect_guard']),('REFUSED',None,None,None));continue
                logits,legal,_=decided[i]
                self.assertEqual((base['action_index'],row['action_index'],row['target_fallback']),(2,0,False))   # averaging's 2; the target's 0
                self.assertEqual((row['action_index'],row['target_fallback']),scorer.target_argmax(wire(row['policy_target']),legal,logits,wire(row['q_values'])))
                self.assertEqual(row['action_index'],row['action_policy_target'])   # the reported target's argmax (constructor_worlds.root_actions)
                self.assertEqual((base['effect_guard']['choice'],row['effect_guard']['choice']),(2,0))
                self.assertEqual({k:v for k,v in row['effect_guard'].items()if k!='choice'},{k:v for k,v in base['effect_guard'].items()if k!='choice'})
        # without the forced target the reported policy_target follows q here: both operators play averaging's action
        rows,_=self.operator_rows(TARGET,reqs,outcomes)
        self.assertEqual([r['action_index']for r in rows],[2,2,2,None]);self.assertEqual([r['target_fallback']for r in rows],[False]*3+[None])
        self.assertEqual([{k:v for k,v in r.items()if k not in(*OPERATOR_KEYS,'pricer')}for r in rows],
            [{k:v for k,v in r.items()if k not in(*OPERATOR_KEYS,'pricer')}for r in plain])   # then whole responses agree but for the operator

    def test_target_argmax_falls_back_to_root_logits_plus_q_without_a_reported_target(self):
        """A reduction that refuses leaves policy_target None: TARGET_ARGMAX plays argmax(root_logits + q_values), target_fallback
        true, as the scorer's target_argmax does on that response."""
        from scripts import depthn_rr_score as scorer
        with patch.object(live,'reported',return_value=dict(reduction_note='REDUCTION_REFUSED: a block reduce_completed_q refused')):
            rows,decided=self.operator_rows(TARGET,[request('f',0)],{('f',0):self.outcome(0)})
        row,=rows;(logits,legal,_),=decided
        self.assertEqual((row['status'],row['policy_target'],row['target_fallback'],row['reduction_note']),
            ('COMPLETE',None,True,'REDUCTION_REFUSED: a block reduce_completed_q refused'))
        acts=[a for a in range(9)if legal[a]]
        self.assertEqual(row['action_index'],max(acts,key=lambda a:(float(logits[a])+row['q_values'][a],-a)))
        self.assertEqual((row['action_index'],True),scorer.target_argmax(wire(row['policy_target']),legal,logits,wire(row['q_values'])))

    def test_guard_c_is_evaluated_on_the_target_argmax_choice(self):
        """The effect guard's (c), our chosen move being guarded for us, is evaluated on the action TARGET_ARGMAX plays: the
        search passes its played action_index to the record (choice). Our lane 0 is Counter (tier A), its root prior forced
        under the turn-1 threshold so (b) does not trip: when the target plays lane 0, (c) alone trips the record, the live
        actor plays greedy and the RR scorer's per-operator guard does too; when it plays lane 1, nothing trips and both play
        it. Averaging's own choice (lane 2) would not have tripped (c): the record follows the operator's choice, not averaging's."""
        from scripts import depthn_rr_score as scorer
        build=cw.build
        def low_counter_prior(*a,**k):   # our root prior on the Counter lane far under THRESHOLD: (b) at turn 1 does not trip
            made=build(*a,**k)
            for root in made.roots:
                if root.get('root_logits')is not None:root['root_logits']=tuple(-30. if lane==0 else float(x)for lane,x in enumerate(root['root_logits']))
            return made
        own=dict(active=[dict(moves=[dict(id='counter'),dict(id='tackle'),dict(id='rest')])])
        line=request('c',0,last=own)['own_request_line'];moves=actor.own_moves(line);greedy=1
        self.assertEqual(moves,('counter','tackle','rest',None))
        for target,c in((0,True),(1,False)):
            with self.subTest(target=target),patch.object(live.cw,'build',side_effect=low_counter_prior):
                (row,),_=self.operator_rows(TARGET,[request('c',0,last=own)],{('c',0):self.outcome(0)},target=target)
            with self.subTest(target=target):
                record=row['effect_guard'];played=row['action_index']
                self.assertEqual((row['status'],played,record['choice'],record['b'],record['c'],record['tripped']),('COMPLETE',target,target,False,c,c))
                self.assertEqual(record['arms'],['choice']if c else[])
                self.assertEqual(G.trips(record,played)['tripped'],c);self.assertFalse(G.trips(record,2)['tripped'])   # averaging's lane 2: clear
                response=wire(row);live_action,cause,guards,_=actor.guarded_action(response,NS(legal_action_mask=CW.T.ROOT_LEGAL),moves,greedy)
                live_played=greedy if live_action is None else live_action
                choices,causes=scorer.guard_choices(dict(target_argmax=played),greedy,list(moves),response['effect_guard'])
                self.assertEqual(live_played,greedy if c else target);self.assertEqual(choices['target_argmax'],live_played)
                self.assertEqual((guards,causes['target_argmax']),(['UNREPRESENTED_EFFECT'],['UNREPRESENTED_EFFECT'])if c else([],None))

    # NOISE_GATED_K2 (screen-r4b) and the default against 3bd60120 ------------------------------------------------------------------
    def test_the_default_operator_is_byte_identical_to_3bd60120(self):
        """The default operator's decide outputs and whole responses (default_corpus) hash to what the same code gives on the
        3bd60120 tree, before any root operator existed."""
        corpus=default_corpus(self)
        self.assertEqual({k:hashlib.sha256('\n'.join(v).encode()).hexdigest()for k,v in corpus.items()},DEFAULT_CORPUS_3BD60120)
        self.assertEqual({k:len(v)for k,v in corpus.items()},dict(decide=720,responses=8))
        self.assertTrue(all('noise_gate'not in json.loads(r)and 'target_fallback'not in json.loads(r)for r in corpus['responses']))

    def gated_rows(self,operator,values,logits,*,last=None):
        """One root's response (stage timings 0) under `operator`, its worlds' prices rewritten: every legal cell of our action a in
        world w (block k's world j is world 2k+j) is values(w, a), whatever the reply; its root logits `logits`. Also the decide
        call's (root_logits, root_legal) and the backed-up worlds decide saw (block order)."""
        backup,build=backups.backup,cw.build;seen=[]
        def rewritten(frame,row,rule):
            pair=backup(frame,row,rule);k=row%4
            out=tuple(original.WorldPrices(p.opponent_requested,p.opponent_legal,p.opponent_logits,
                tuple(tuple(None if v is None else values(2*k+j,a)for v in p.joint_values[a])for a in range(9)))for j,p in enumerate(pair))
            seen.extend(out);return out
        def logits_of(*a,**k):
            made=build(*a,**k)
            for root in made.roots:
                if root.get('root_logits')is not None:root['root_logits']=tuple(map(float,logits))
            return made
        with patch.object(live.backups,'backup',side_effect=rewritten),patch.object(live.cw,'build',side_effect=logits_of):
            rows,decided=self.operator_rows(operator,[request('g',0,last=last)],{('g',0):self.outcome(0)})
        return rows[0],decided[0],tuple(seen)

    def test_noise_gated_k2_plays_the_scorers_noise_gated_and_changes_only_the_played_action(self):
        """The same root under the default and NOISE_GATED_K2, in each regime of the gate (W 8; averaging plays lane 2 by
        construction; the root logits' greedy g is lane 1, or lane 2 itself): a* == g (nothing to gate: g plays, passed false);
        a constant per-world gap (sd 0, mean > 0: the gate passes, a* plays); a gap whose mean is under 2 standard errors (g
        plays). The NOISE_GATED_K2 response names its operator (its pricer identity too) and adds noise_gate; its played action
        is the RR scorer's own noise_gated(per_world_q(grids), a*, argmax root_logits, 2) on the worlds decide saw; everything
        else is the default's, the effect guard record but its `choice` (the played action) included."""
        from scripts import depthn_rr_score as scorer
        noisy=lambda w,a:{2:.5 if w%2 else -.45,1:0.}.get(a,-.9)   # gaps +.5/-.45: mean .025, 2 SE about .34: the gate fails
        flat=lambda w,a:{2:.6,1:.1}.get(a,-.9)                       # gap .5 in every world: sd 0, the gate passes
        for name,values,logits,want in(('a* == g',flat,(0.,0.,3.)+(0.,)*6,(2,False,2)),('passes',flat,(0.,3.,0.)+(0.,)*6,(2,True,1)),
                ('fails',noisy,(0.,3.,0.)+(0.,)*6,(1,False,1))):
            with self.subTest(name):
                base,_,_=self.gated_rows(live.ROOT_OPERATOR,values,logits);row,(root_logits,legal,kw),seen=self.gated_rows(NOISE,values,logits)
                encode(base);encode(row);self.assertEqual(kw,dict(operator=NOISE))
                self.assertEqual(set(base),KEYS);self.assertEqual(set(row),KEYS|{'noise_gate'})
                self.assertEqual((base['status'],base['action_index'],row['status']),('COMPLETE',2,'COMPLETE'))
                self.assertEqual((row['action_index'],row['noise_gate']['passed'],row['noise_gate']['greedy']),want)
                self.assertEqual(row['noise_gate']['candidate'],2)
                self.assertEqual((row['root_operator'],row['root_operator_impl'],row['pricer']['root_operator'],row['pricer']['root_operator_impl']),
                    (NOISE,'depthn_root_operator.noise_gated',NOISE,'depthn_root_operator.noise_gated'))
                same=lambda r:{k:v for k,v in r.items()if k not in(*NOISE_KEYS,'pricer','action_index','effect_guard')}
                self.assertEqual(same(row),same(base))
                self.assertEqual({k:v for k,v in row['effect_guard'].items()if k!='choice'},{k:v for k,v in base['effect_guard'].items()if k!='choice'})
                self.assertEqual(row['effect_guard']['choice'],row['action_index'])
                gs=hedge.grids(seen,legal);greedy=scorer.argmax(list(root_logits),[a for a in range(9)if legal[a]])
                self.assertEqual(row['action_index'],scorer.noise_gated(scorer.per_world_q(gs),base['action_index'],greedy,2))
                self.assertEqual(wire(row['noise_gate']),dict(passed=row['action_index']!=greedy,greedy=greedy,candidate=base['action_index']))
        rows,_=self.operator_rows(NOISE,[request('r',0)],{('r',0):Refusal('SEARCH_PREFIX_OR_REQUEST_MISMATCH')})
        self.assertEqual((rows[0]['status'],rows[0]['action_index'],rows[0]['noise_gate'],rows[0]['effect_guard']),('REFUSED',None,None,None))

    def test_guard_c_is_evaluated_on_the_noise_gated_choice(self):
        """The effect guard's (c) is evaluated on the action NOISE_GATED_K2 plays. Our lane 0 is Counter (tier A, its root prior far
        under the turn-1 threshold, so (b) does not trip), the root logits' greedy is lane 1 (Tackle), averaging's a* lane 0:
        when the gate passes, Counter plays and (c) alone trips the record, so the live actor and the RR scorer's per-operator
        guard both play greedy; when it fails, lane 1 plays, nothing trips and both play it."""
        from scripts import depthn_rr_score as scorer
        own=dict(active=[dict(moves=[dict(id='counter'),dict(id='tackle'),dict(id='rest')])])
        line=request('c',0,last=own)['own_request_line'];moves=actor.own_moves(line);greedy=1;logits=(-30.,3.,0.)+(0.,)*6
        flat=lambda w,a:{0:.6,1:.1}.get(a,-.9);noisy=lambda w,a:{0:.5 if w%2 else -.45,1:0.}.get(a,-.9)
        for name,values,played,c in(('passes',flat,0,True),('fails',noisy,1,False)):
            with self.subTest(name):
                row,_,_=self.gated_rows(NOISE,values,logits,last=own);record=row['effect_guard']
                self.assertEqual((row['status'],row['action_index'],row['noise_gate']['candidate'],row['noise_gate']['passed']),('COMPLETE',played,0,c))
                self.assertEqual((record['choice'],record['b'],record['c'],record['tripped']),(played,False,c,c))
                self.assertEqual(G.trips(record,0)['c'],True);self.assertFalse(G.trips(record,1)['tripped'])   # a* would trip; g would not
                response=wire(row);live_action,cause,guards,_=actor.guarded_action(response,NS(legal_action_mask=CW.T.ROOT_LEGAL),moves,greedy)
                live_played=greedy if live_action is None else live_action
                choices,causes=scorer.guard_choices({'noise_gated:k=2':played},greedy,list(moves),response['effect_guard'])
                self.assertEqual(live_played,greedy if c else played);self.assertEqual(choices['noise_gated:k=2'],live_played)
                self.assertEqual((guards,causes['noise_gated:k=2']),(['UNREPRESENTED_EFFECT'],['UNREPRESENTED_EFFECT'])if c else([],None))


def world(values,*,opponent=None,logits=None):
    """WorldPrices with the given action -> {reply: value} rows; `opponent` the legal replies (None: a waiting opponent, column 0)."""
    jv=[[None]*9 for _ in range(9)]
    for a,row in values.items():
        for b,v in row.items():jv[a][b]=v
    legal=tuple(b in opponent for b in range(9))if opponent else(False,)*9
    return original.WorldPrices(bool(opponent),legal,tuple(logits or(0.,)*9),tuple(map(tuple,jv)))


class AveragingEquivalenceTests(unittest.TestCase):
    """The played action against the deployment cell's search_hedge.averaging (v2_search_head_to_head.Search.choose:
    `average = hedge.averaging(prices, root_legal)`, prices = every block's backed-up worlds concatenated in block order; the
    'average' cell plays its one-hot strategy) on the SAME per-world prices, and the reported action_mean_q against both.

    The map. Block k (k = 0..K-1, K = W/2) prices worlds 2k and 2k+1 (constructor_worlds.build: request row k holds the pair
    worlds[2k:2k+2]) and depthn_backups.backup(frame, row_k, learner_max) gives its two WorldPrices (at N=2 the numbers of
    depth2_backups.backup_learner_max, the cell's backup). World j of block k is averaging's world 2k+j. decide() calls
    averaging on exactly that concatenation, so the played action and q_values are averaging's by construction, ties
    included; this class checks it on every case, the pinned ones too.

    The reported action_mean_q (constructor_worlds' composition: reduce_completed_q per block, q_k(a) = fsum_{w in k, b}
    t_wb / 2, then Q(a) = fsum_k q_k(a) / K, with t_wb = fl(pi_w(b) G_w[a][b]) as averaging's own products, averaging's
    being Q'(a) = fsum_{w, b} t_wb / W): W, K and 2 are powers of two, so each division is exact (barring subnormals) and
    W Q(a) = fl(sum_k fl(S_k)), W Q'(a) = fl(sum_k S_k), S_k = sum_{w in k, b} t_wb: they differ only by the rounding of the
    K block sums, |Q - Q'| <= 3u (u = 2^-53). So action_mean_q equals the played action unless averaging's best two legal
    actions are within ~6u and group differently across blocks (pinned below, both directions; in the random sweep only
    with artificially equal reply priors and discrete values); mean_q_agrees reports it. reduce_completed_q also raises
    when a block's completed-Q rounds outside [-1, 1] (an all-win block, its prior summing to 1 + ulp); averaging has no
    such check, so that root still plays and only the reported fields are None (reduction_note).
    """
    def played(self,prices,legal,logits=(0.,)*9):
        return live.decide([tuple(prices[2*k:2*k+2])for k in range(len(prices)//2)],tuple(logits),tuple(legal))

    def theirs(self,prices,legal):
        choice=hedge.averaging(tuple(prices),tuple(legal))
        if choice is None:return None,None
        action=hedge.top_action(choice.strategy);self.assertEqual(hedge.sample(choice.strategy,.999999),action)   # the cell's draw
        return action,choice.q

    def reduced(self,prices,legal,logits=(0.,)*9):
        """constructor_worlds.reduce itself on one root whose block k backs up to prices[2k:2k+2] (its learner_max rule checked)."""
        K=len(prices)//2;rules=[]
        def backup(frame,row,rule):rules.append(rule);return tuple(prices[2*row:2*row+2])
        root=dict(index=0,request_id='r',status='PRICED',reason=None,rows=tuple(range(K)),root_logits=tuple(logits),root_legal=tuple(legal))
        with patch.object(cw.backups,'backup',side_effect=backup),patch.object(cw.backups,'refusal_summary',return_value=dict(roots=[dict(cause=None)]*K)):
            result=cw.reduce(object(),cw.Batch(tuple(range(K)),None,(root,),2*K,K))[0]
        self.assertTrue(all(r is backups.learner_max for r in rules))
        return result

    def check(self,prices,legal,logits=(0.,)*9):
        """The played decision is averaging's; the reported fields are constructor_worlds.reduce's (when it labels)."""
        decision=self.played(prices,legal,logits);action,q=self.theirs(prices,legal)
        self.assertEqual((decision['status'],decision['action_index'],decision['q_values']),('COMPLETE',action,list(q)))
        if decision['reduction_note']is None:
            reduced=self.reduced(prices,legal,logits);self.assertEqual(reduced['status'],'LABELED')
            self.assertEqual((decision['action_mean_q'],decision['completed_q'],decision['policy_target']),
                (reduced['action_mean_q'],list(reduced['completed_q']),list(reduced['policy_target'])))
            self.assertEqual(decision['mean_q_agrees'],reduced['action_mean_q']==action)
            self.assertLessEqual(max(abs(x-y)for x,y in zip(decision['completed_q'],q)if x is not None),3*U)
        return decision

    def random_world(self,rng,legal,mode):
        requested=rng.random()<.8;opponent=[b for b in range(9)if rng.random()<.5]if requested else None
        if requested and not opponent:opponent=[int(rng.integers(9))]
        value={'f64':lambda:float(rng.uniform(-1,1)),'f32':lambda:float(np.float32(rng.uniform(-1,1))),
            'terminal':lambda:float(rng.choice([-1.,0.,1.])),'ties':lambda:float(rng.choice([-1.,0.,1.]))}[mode]
        logits=(0.,)*9 if mode=='ties'else[float(x)for x in rng.normal(0,2,9)]   # 'ties': equal reply priors (1/n), the tie-rich regime
        return world({a:{b:value()for b in(opponent or[0])}for a in range(9)if legal[a]},opponent=opponent,logits=logits)

    def test_random_roots_play_averagings_action_and_report_when_mean_q_differs(self):
        rng=np.random.default_rng(20261108);differ={};notes={}
        for mode,cases in(('f64',300),('f32',300),('terminal',300),('ties',1500)):
            for W in(8,16):
                differ[mode,W]=notes[mode,W]=0
                for case in range(cases):
                    legal=tuple(bool(x)for x in rng.random(9)<.6)
                    if not any(legal):legal=(True,)+legal[1:]
                    with self.subTest(mode=mode,W=W,case=case):
                        decision=self.check([self.random_world(rng,legal,mode)for _ in range(W)],legal)
                        differ[mode,W]+=decision['mean_q_agrees']is False;notes[mode,W]+=decision['reduction_note']is not None
        self.assertEqual({k:v for k,v in differ.items()if k[0]!='ties'},{(m,W):0 for m in('f64','f32','terminal')for W in(8,16)})
        self.assertGreater(differ['ties',8]+differ['ties',16],0)   # the regime where it happens: reported, never played
        self.assertEqual(sum(notes.values()),sum(notes[k]for k in notes if k[0]in('terminal','ties')))

    def test_exact_ties_go_to_the_lowest_index(self):
        rng=np.random.default_rng(7);legal=(False,True,True,False,True,True,False,True,False)
        for W in(8,16):
            # identical columns: actions 2 and 5 share every world's values (continuous), every other action is lower
            prices=[]
            for w in range(W):
                opponent=sorted(rng.choice(9,size=3,replace=False).tolist());top={b:float(rng.uniform(.5,1))for b in opponent}
                rows={a:({b:v for b,v in top.items()}if a in(2,5)else{b:float(rng.uniform(-1,.4))for b in opponent})for a in range(9)if legal[a]}
                prices.append(world(rows,opponent=opponent,logits=[float(x)for x in rng.normal(0,1,9)]))
            decision=self.check(prices,legal);self.assertEqual((decision['action_index'],decision['action_mean_q']),(2,2))
            self.assertEqual(decision['q_values'][2],decision['q_values'][5])
            # exactly representable sums placed in different blocks: dyadic values, unit or 1/2-1/2 priors
            prices=[world({a:{0:0.}for a in range(9)if legal[a]})for _ in range(W)]
            prices[0]=world({1:{3:.5,6:.25},2:{3:0.,6:0.},4:{3:.125,6:0.},5:{3:0.,6:0.},7:{3:-.5,6:.5}},opponent=[3,6])
            prices[W-1]=world({1:{0:0.},2:{0:.375},4:{0:.25},5:{0:0.},7:{0:-.25}})
            decision=self.check(prices,legal)
            self.assertEqual(decision['q_values'][1],decision['q_values'][2]);self.assertEqual(decision['completed_q'],decision['q_values'])
            self.assertEqual((decision['action_index'],decision['action_mean_q'],decision['mean_q_agrees']),(1,1,True))
            same=[world({a:{0:.3}for a in range(9)if legal[a]})for _ in range(W)]   # every legal action tied: the lowest legal index
            self.assertEqual(self.check(same,legal)['action_index'],1)

    def test_one_ulp_near_ties_play_averagings_action_and_report_the_disagreement(self):
        legal=(True,True)+(False,)*7;e=2.**-55
        pair=lambda a0,a1:[world({0:{0:x},1:{0:y}})for x,y in zip(a0,a1)]
        # averaging resolves 0.5 + 2^-53 for action 1; the block sums (0.5 + 2^-54 per block, half an ulp) round to even: mean-Q ties
        decision=self.check(pair([.5]+[0.]*7,[.5,2.*e,2.*e]+[0.]*5),legal)
        self.assertEqual((decision['action_index'],decision['action_mean_q'],decision['mean_q_agrees']),(1,0,False))
        self.assertEqual(decision['q_values'][1]-decision['q_values'][0],2.**-56);self.assertEqual(decision['completed_q'][0],decision['completed_q'][1])
        # the other way: three blocks each round up by 2^-55, the exact sums tie (averaging plays 0), mean-Q prefers 1 by one ulp
        decision=self.check(pair([.75]+[0.]*7,[.25,3*e,.25,3*e,.25,3*e,-9*e,0.]),legal)
        self.assertEqual((decision['action_index'],decision['action_mean_q'],decision['mean_q_agrees']),(0,1,False))
        self.assertEqual(decision['q_values'][0],decision['q_values'][1]);self.assertEqual(decision['completed_q'][1]-decision['completed_q'][0],2.**-56)

    def test_unknown_prices_refuse_and_an_out_of_range_block_only_blanks_the_report(self):
        from gpu_engine.d3_search_latency import reduce_block
        legal=(True,True)+(False,)*7;base=[world({0:{0:.1},1:{0:.2}})for _ in range(8)]
        missing=list(base);missing[5]=world({0:{0:.1},1:{0:None}})
        self.assertEqual(self.theirs(missing,legal)[0],None)
        self.assertEqual((self.played(missing,legal)['status'],self.played(missing,legal)['refusal']),('REFUSED','AVERAGING_UNKNOWN_PRICE'))
        logits=[1.746,-2.561,-1.426]+[0.]*6;wins=world({0:{0:1.,1:1.,2:1.},1:{0:0.,1:0.,2:0.}},opponent=[0,1,2],logits=logits)   # its prior sums to 1 + ulp
        prices=[wins,wins]+base[2:]
        with self.assertRaisesRegex(ValueError,'in-range'):reduce_block((wins,wins),(0.,)*9,legal,original)   # block 0's completed-Q is 1 + ulp
        decision=self.check(prices,legal)
        self.assertEqual((decision['action_index'],decision['reduction_note']),(0,'REDUCTION_ERROR: ValueError: finite in-range numerical value required'))
        self.assertEqual([decision[k]for k in('completed_q','policy_target','root_value','action_mean_q','action_policy_target','mean_q_agrees')],[None]*6)

    # TARGET_ARGMAX ---------------------------------------------------------------------------------------------------------------
    def target_check(self,prices,legal,logits):
        """decide under TARGET_ARGMAX against the default on the same root: every field but action_index the default's (plus
        target_fallback), and (action_index, target_fallback) the RR scorer's target_argmax on the decision's own policy_target
        and q_values as the wire carries them, with the root's legal mask and logits."""
        from scripts import depthn_rr_score as scorer
        blocks=[tuple(prices[2*k:2*k+2])for k in range(len(prices)//2)]
        base=live.decide(blocks,tuple(logits),tuple(legal));mine=live.decide(blocks,tuple(logits),tuple(legal),operator=TARGET)
        self.assertNotIn('target_fallback',base);self.assertEqual(set(mine),set(base)|{'target_fallback'})
        self.assertEqual({k:v for k,v in mine.items()if k not in('action_index','target_fallback')},{k:v for k,v in base.items()if k!='action_index'})
        want=scorer.target_argmax(wire(mine['policy_target']),tuple(legal),tuple(logits),wire(mine['q_values']))
        self.assertEqual((mine['action_index'],mine['target_fallback']),want)
        self.assertIs(scorer.target_argmax,root_ops.target_argmax);self.assertIs(scorer.argmax,root_ops.argmax)   # one definition
        return base,mine

    def test_target_argmax_is_the_scorers_function_on_random_roots(self):
        """Random roots (every value mode, W 8 and 16, random root logits): the TARGET_ARGMAX decision is the scorer's
        target_argmax, which with a reported target is that target's argmax (action_policy_target); ties in the target, the
        tie-rich 'ties' mode's equal targets, go to the lowest index."""
        rng=np.random.default_rng(20261109);overrides=ties=0
        for mode,cases in(('f64',150),('f32',150),('terminal',150),('ties',400)):
            for W in(8,16):
                for case in range(cases):
                    legal=tuple(bool(x)for x in rng.random(9)<.6)
                    if not any(legal):legal=(True,)+legal[1:]
                    logits=(0.,)*9 if mode=='ties'else tuple(float(x)for x in rng.normal(0,1,9))
                    with self.subTest(mode=mode,W=W,case=case):
                        base,mine=self.target_check([self.random_world(rng,legal,mode)for _ in range(W)],legal,logits)
                        if mine['status']!='COMPLETE'or mine['target_fallback']:continue
                        acts=[a for a in range(9)if legal[a]];target=mine['policy_target']
                        self.assertEqual(mine['action_index'],mine['action_policy_target'])
                        top=max(target[a]for a in acts);tied=[a for a in acts if target[a]==top]
                        self.assertEqual(mine['action_index'],tied[0]);ties+=len(tied)>1;overrides+=mine['action_index']!=base['action_index']
        self.assertGreater(ties,0);self.assertGreater(overrides,0)   # both regimes are exercised

    def test_target_argmax_falls_back_on_a_missing_or_nonfinite_target_ties_included(self):
        """No reported target (the reduction refused or raised: policy_target None) or a non-finite one on a legal action: the
        scorer's fallback, argmax(root_logits + q), lowest index on ties."""
        rng=np.random.default_rng(11);fallbacks=ties=0
        with patch.object(live,'reported',return_value=dict(reduction_note='REDUCTION_REFUSED: a block reduce_completed_q refused')):
            for mode,cases in(('f64',100),('ties',300)):
                for case in range(cases):
                    legal=tuple(bool(x)for x in rng.random(9)<.6)
                    if not any(legal):legal=(True,)+legal[1:]
                    logits=(0.,)*9 if mode=='ties'else tuple(float(x)for x in rng.normal(0,1,9))
                    with self.subTest(mode=mode,case=case):
                        base,mine=self.target_check([self.random_world(rng,legal,mode)for _ in range(8)],legal,logits)
                        if mine['status']!='COMPLETE':continue
                        self.assertTrue(mine['target_fallback']);fallbacks+=1
                        acts=[a for a in range(9)if legal[a]];score={a:logits[a]+mine['q_values'][a]for a in acts};top=max(score.values())
                        tied=[a for a in acts if score[a]==top];self.assertEqual(mine['action_index'],tied[0]);ties+=len(tied)>1
        self.assertGreater(fallbacks,0);self.assertGreater(ties,0)
        # the out-of-range block (REDUCTION_ERROR): no target, so root_logits + q decide; logits favouring action 1 override averaging's 0
        legal=(True,True)+(False,)*7;base_worlds=[world({0:{0:.1},1:{0:.2}})for _ in range(8)]
        wins=world({0:{0:1.,1:1.,2:1.},1:{0:0.,1:0.,2:0.}},opponent=[0,1,2],logits=[1.746,-2.561,-1.426]+[0.]*6)
        base,mine=self.target_check([wins,wins]+base_worlds[2:],legal,(0.,5.)+(0.,)*7)
        self.assertEqual((base['action_index'],mine['action_index'],mine['target_fallback'],mine['policy_target']),(0,1,True,None))
        self.assertTrue(mine['reduction_note'].startswith('REDUCTION_ERROR'))
        # a non-finite target on a legal action: the scorer's rule falls back too (such a response never passes encode)
        from gpu_engine import d3_search_latency as latency
        compose=latency.compose
        nan=float('nan')   # one object, so the default's and TARGET_ARGMAX's reported targets compare equal
        def nan_target(*a):out=compose(*a);return dict(out,policy_target=(nan,)+tuple(out['policy_target'][1:]))
        prices=[world({0:{0:.1},1:{0:.3}})for _ in range(8)]
        with patch.object(latency,'compose',side_effect=nan_target):base,mine=self.target_check(prices,legal,(2.,0.)+(0.,)*7)
        self.assertEqual((base['action_index'],mine['action_index'],mine['target_fallback']),(1,0,True))   # 2 + .1 > 0 + .3
        with self.assertRaisesRegex(ValueError,'root operator'):live.decide([tuple(prices[:2])],(0.,)*9,legal,operator='ARGMAX_POLICY_TARGET')

    # NOISE_GATED_K2 ----------------------------------------------------------------------------------------------------------------
    def test_noise_gated_k2_is_the_scorers_function_on_random_roots(self):
        """1,200 random roots (W 2, 4, 8, 16; every value mode, plus 'clone': one world repeated W times, so every per-world gap is
        the same and sd = 0; root logits random, all zero (g the lowest legal index) or on averaging's own action (a* == g)): the
        NOISE_GATED_K2 decision is the default's but for action_index and noise_gate, and its action is the RR scorer's
        noise_gated:k=2 from depthn_rr_score.operators itself, on the worlds as a re-pricer row holds them (wire form, through
        the scorer's recompute) with greedy = the scorer's greedy_argmax of the root logits. Every regime occurs: ties in q
        (lowest index), a* == g, the gate passing and failing, sd = 0 with a positive and with a zero mean gap (the strict
        comparison: g plays), W < 16."""
        from scripts import depthn_rr_score as scorer
        self.assertIs(scorer.noise_gated,root_ops.noise_gated);self.assertIs(scorer.per_world_q,root_ops.per_world_q)   # one definition
        self.assertIn('noise_gated:k=2',scorer.SETTINGS);self.assertEqual(root_ops.NOISE_K2,2)
        rng=np.random.default_rng(20261110);seen=collections.Counter();failures=collections.Counter()
        def wire_world(p):return wire(dict(requested=p.opponent_requested,kept=list(p.opponent_legal),logits=list(p.opponent_logits),
            grid=[list(r)for r in p.joint_values]))
        for mode,cases in(('f64',60),('f32',60),('terminal',60),('ties',60),('clone',60)):
            for W in(2,4,8,16):
                for case in range(cases):
                    legal=tuple(bool(x)for x in rng.random(9)<.6)
                    if not any(legal):legal=(True,)+legal[1:]
                    if mode=='clone':prices=[self.random_world(rng,legal,('f64','ties')[case%2])]*W
                    else:prices=[self.random_world(rng,legal,mode)for _ in range(W)]
                    blocks=[tuple(prices[2*k:2*k+2])for k in range(W//2)]
                    kind=case%3;acts=[a for a in range(9)if legal[a]]
                    if kind==0:logits=tuple(float(x)for x in rng.normal(0,1,9))
                    elif kind==1:logits=(0.,)*9
                    else:
                        base=live.decide(blocks,(0.,)*9,legal)
                        logits=tuple(4. if a==base['action_index']else 0. for a in range(9))if base['status']=='COMPLETE'else(0.,)*9
                    with self.subTest(mode=mode,W=W,case=case):
                        base=live.decide(blocks,logits,legal);mine=live.decide(blocks,logits,legal,operator=NOISE)
                        self.assertNotIn('noise_gate',base);self.assertEqual(set(mine),set(base)|{'noise_gate'})
                        self.assertEqual({k:v for k,v in mine.items()if k not in('action_index','noise_gate')},{k:v for k,v in base.items()if k!='action_index'})
                        if base['status']!='COMPLETE':self.assertIsNone(mine['noise_gate']);seen['refused']+=1;continue
                        decision=dict(replay_status='COMPLETE',root_legal=list(legal),root_logits=wire(list(logits)),worlds=[wire_world(p)for p in prices])
                        b=scorer.recompute(decision);greedy=scorer.argmax(decision['root_logits'],acts)
                        ops,_=scorer.operators(b.worlds,b.grids,b.average,greedy,b.logits,b.legal,None,failures)
                        chosen=base['action_index']
                        self.assertEqual(hedge.top_action(b.average.strategy),chosen)
                        self.assertEqual(mine['action_index'],ops['noise_gated:k=2'])
                        self.assertEqual(mine['noise_gate'],dict(passed=mine['action_index']!=greedy,greedy=greedy,candidate=chosen))
                        q=base['q_values'];top=max(q[a]for a in acts);seen['q_tie']+=sum(q[a]==top for a in acts)>1
                        seen[f'W{W}']+=1
                        if chosen==greedy:seen['a*==g']+=1;self.assertEqual(mine['action_index'],greedy);continue
                        gaps=[x[chosen]-x[greedy]for x in scorer.per_world_q(b.grids)]
                        seen['passed'if mine['noise_gate']['passed']else'failed']+=1
                        if len(set(gaps))==1:
                            seen['sd0_positive'if gaps[0]>0 else'sd0_zero'if gaps[0]==0 else'sd0_negative']+=1
                            self.assertEqual(mine['action_index'],chosen if gaps[0]>0 else greedy)
        self.assertEqual(sum(seen[f'W{W}']for W in(2,4,8,16))+seen['refused'],1200)
        for regime in('q_tie','a*==g','passed','failed','sd0_positive','sd0_zero','W2','W4','W8','W16'):
            self.assertGreater(seen[regime],0,regime)
        self.assertEqual(sum(failures.values()),0)

    def test_noise_gated_is_strict_at_the_threshold(self):
        """The scorer's rule on hand-made per-world Q: a mean gap exactly at k standard errors plays greedy (W 2, gaps 3 and 1: mean 2,
        2 sd / sqrt 2 = 2 exactly), just above it averaging's choice; sd 0 with a zero gap greedy, with a positive gap the choice;
        a* == greedy plays greedy without consulting the gate."""
        q=lambda *gaps:[{0:g,1:0.}for g in gaps]
        self.assertEqual(root_ops.noise_gated(q(3.,1.),0,1,2),1)
        self.assertEqual(root_ops.noise_gated(q(3.,math.nextafter(1.,2.)),0,1,2),0)
        self.assertEqual((root_ops.noise_gated(q(0.,0.,0.),0,1,2),root_ops.noise_gated(q(.1,.1,.1),0,1,2)),(1,0))
        self.assertEqual(root_ops.noise_gated(q(-5.,-5.),1,1,2),1)
        self.assertEqual(root_ops.logits_greedy((0.,2.,2.)+(0.,)*6,(True,True,True)+(False,)*6),1)   # lowest index on ties
        self.assertEqual(root_ops.logits_greedy((9.,2.,2.)+(0.,)*6,(False,True,True)+(False,)*6),1)   # over the legal actions only


if __name__=='__main__':unittest.main()
