"""Authored owning-wire fixtures only; no corpus/checkpoint/simulator access."""
from dataclasses import replace
import hashlib
import json
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pokezero.live_protocol_actor import (ActorContract, JaxBackend, LiveProtocolActor,
    MASK_NAMES, NoAction, OwnedProvider, PythonBackend, RawPolicy, SHAPES, _canonical,
    _request_targets, _vocab_sha)
from pokezero.policy import PolicyDecision
from pokezero.showdown import _legal_action_mask


def request(owner='p1', *, wait=False, forced=False, active_index=1):
    value = dict(rqid=3, side=dict(id=owner,pokemon=[dict(ident=f'{owner}: Mon{i}',
        details='Snorlax, L80',condition='120/240',active=i==active_index,
        moves=['bodyslam'],stats=dict(atk=100,def_=100,spa=100,spd=100,spe=100)) for i in range(3)]))
    if wait:
        value['wait'] = True
    elif forced:
        value['forceSwitch'] = [True]
    else:
        value['active'] = [dict(moves=[dict(id='bodyslam',pp=0,maxpp=24),dict(id='surf',disabled=True)])]
    return '|request|'+json.dumps(value)


class FixtureBackend(OwnedProvider):
    def __init__(self, contract):
        super().__init__(contract)
        self.lines, self.encodes, self.closed = [], [], 0
        self.fail_line = self.mutate = None

    def ingest(self, lines):
        for line in lines:
            self.lines.append(line)
            if line == self.fail_line:
                raise ValueError('authored late line failure')

    def encode(self, snapshot):
        self.encodes.append(snapshot)
        packet = json.loads(snapshot.own_request_line[len('|request|'):])
        arrays = {k:np.zeros(s,np.float32 if k=='numeric_features' else bool if k.endswith('mask') else np.int32)
                  for k,s in SHAPES.items()}
        arrays['attention_mask'][:] = True
        arrays['legal_action_mask'][:] = _legal_action_mask(packet)
        if self.mutate:
            self.mutate(arrays)
        return arrays,_request_targets(packet)

    def close(self):
        self.closed += 1


class ActorTests(unittest.TestCase):
    def setUp(self):
        self.contract = ActorContract.champion()
        self.backends, self.observations, self.resets = [], [], []
        self.action = 0
        self.stack_patches = [patch('subprocess.Popen',side_effect=AssertionError('no process')),
                              patch('subprocess.run',side_effect=AssertionError('no runtime'))]
        for p in self.stack_patches:
            p.start(); self.addCleanup(p.stop)

    def make(self, *, backend_factory=None, policy_factory=None):
        def backend(c, **kwargs):
            b = FixtureBackend(c); self.backends.append(b); return b
        def policy(c):
            def choose(obs, *, rng):
                self.observations.append(obs)
                return PolicyDecision(self.action,'authored-raw',value_estimate=.25)
            return RawPolicy(c, choose, lambda:self.resets.append(True))
        return LiveProtocolActor(self.contract,backend_factory=backend_factory or backend,
                                 policy_factory=policy_factory or policy)

    def begin(self, actor, *, owner='p1', battle='authored', stream=1):
        actor.begin_game(battle_id=battle,owner=owner,seed=42,stream_id=stream)

    def feed(self, actor, lines):
        actor.accept_owner_stream(battle_id=actor.battle_id,owner=actor.owner,
            generation=actor.generation,prefix_sha256=actor.prefix_sha256,lines=tuple(lines))

    def decide(self, actor):
        snap=actor.snapshot_boundary()
        result=actor.decide(snap,rng=random.Random(42))
        return snap,result

    def test_literal_interleaving_every_request_wait_and_common_snapshot(self):
        actor=self.make(); self.begin(actor)
        lines=('|turn|1',request(wait=True),'|-message|literal owner hint',request(), '|')
        self.feed(actor,lines)
        snap,decision=self.decide(actor)
        self.assertEqual(snap.lines,lines); self.assertEqual(snap.request_count,2)
        self.assertEqual(self.backends[0].lines,list(lines))
        self.assertIs(self.backends[0].encodes[0],snap)
        self.assertEqual(snap.prefix_sha256,hashlib.sha256(''.join(s+'\n' for s in lines).encode()).hexdigest())
        self.assertEqual(actor.choice(snap),'move 1'); self.assertEqual(decision.action_index,0)
        self.assertEqual(self.observations[0].metadata['owner_prefix_sha256'],snap.prefix_sha256)
        self.assertEqual(self.observations[0].perspective.player_id,'p1')
        with self.assertRaises(TypeError):self.observations[0].metadata['generation']=100

    def test_decision_observation_is_exact_policy_input_without_reexecution(self):
        actor=self.make(); self.begin(actor); self.feed(actor,(request(),))
        snap=actor.snapshot_boundary()
        with self.assertRaisesRegex(ValueError,'no completed'):actor.decision_observation(snap)
        self.assertEqual(self.backends[-1].encodes,[])
        actor.decide(snap,rng=random.Random(42))
        observation=actor.decision_observation(snap)
        self.assertIs(observation,self.observations[-1])
        self.assertIs(actor.decision_observation(snap),observation)
        self.assertEqual(len(self.backends[-1].encodes),1)
        self.assertEqual(len(self.observations),1)
        with self.assertRaisesRegex(ValueError,'stale'):
            actor.decision_observation(replace(snap))
        self.assertIs(actor.decision_observation(snap),observation)
        with self.assertRaisesRegex(ValueError,'already consumed'):actor.decide(snap,rng=random.Random())
        with self.assertRaisesRegex(ValueError,'no completed'):actor.decision_observation(snap)
        self.assertEqual(len(self.backends[-1].encodes),1)

    def test_decision_observation_primitives_and_metadata_are_immutable(self):
        from dataclasses import FrozenInstanceError
        actor=self.make(); self.begin(actor); self.feed(actor,(request(),))
        snap,_=self.decide(actor); observation=actor.decision_observation(snap)
        for name in SHAPES:self.assertIs(type(getattr(observation,name)),tuple)
        self.assertIs(type(observation.numeric_features[0]),tuple)
        self.assertIs(type(observation.numeric_features[0][0]),float)
        self.assertIs(type(observation.categorical_ids[0][0]),int)
        self.assertIs(type(observation.attention_mask[0]),bool)
        with self.assertRaises(TypeError):observation.numeric_features[0][0]=5.
        with self.assertRaises(TypeError):observation.metadata['generation']=999
        with self.assertRaises(FrozenInstanceError):observation.numeric_features=()
        self.assertTrue(all(type(v) in (str,int,bool) for v in observation.metadata.values()))

    def test_decision_observation_cleared_on_newstream_poison_close_and_reset(self):
        actor=self.make(); self.begin(actor); self.feed(actor,(request(),))
        previous,_=self.decide(actor)
        self.feed(actor,(request(),))
        with self.assertRaises(ValueError):actor.decision_observation(previous)
        current=actor.snapshot_boundary()
        with self.assertRaisesRegex(ValueError,'no completed'):actor.decision_observation(current)
        self.action=1
        with self.assertRaisesRegex(ValueError,'illegal'):actor.decide(current,rng=random.Random())
        with self.assertRaises(ValueError):actor.decision_observation(current)
        self.assertEqual(actor.close_game()['status'],'FAILED_CLOSED')
        with self.assertRaises(ValueError):actor.decision_observation(current)
        self.begin(actor,battle='next',stream=2)
        with self.assertRaises(ValueError):actor.decision_observation(current)
        with self.assertRaisesRegex(ValueError,'no completed'):
            actor.decision_observation(actor.snapshot_boundary())

    def test_decision_observation_refuses_terminal_wait_and_reissued_snapshot(self):
        actor=self.make(); self.begin(actor); self.feed(actor,(request(wait=True),))
        snap=actor.snapshot_boundary()
        with self.assertRaises(NoAction):actor.decide(snap,rng=random.Random())
        with self.assertRaisesRegex(ValueError,'no completed'):actor.decision_observation(snap)
        self.feed(actor,(request(),)); snap,_=self.decide(actor)
        new=actor.snapshot_boundary()
        with self.assertRaises(ValueError):actor.decision_observation(snap)
        with self.assertRaisesRegex(ValueError,'no completed'):actor.decision_observation(new)
        self.feed(actor,('|win|Authored',)); terminal=actor.snapshot_boundary()
        with self.assertRaises(NoAction):actor.decide(terminal,rng=random.Random())
        with self.assertRaisesRegex(ValueError,'no completed'):actor.decision_observation(terminal)
        self.assertEqual(len(self.backends[-1].encodes),1)

    def test_decision_observation_not_published_when_late_policy_guard_fails(self):
        holder={}
        def choose(observation,*,rng):
            self.observations.append(observation)
            holder['policy'].contract=replace(self.contract,checkpoint_sha256='f'*64)
            return PolicyDecision(0,'authored')
        def policy(c):
            holder['policy']=RawPolicy(c,choose,lambda:None)
            return holder['policy']
        actor=self.make(policy_factory=policy); self.begin(actor); self.feed(actor,(request(),))
        snap=actor.snapshot_boundary()
        with self.assertRaisesRegex(ValueError,'namespace drift'):actor.decide(snap,rng=random.Random())
        with self.assertRaisesRegex(ValueError,'not active'):actor.decision_observation(snap)
        self.assertEqual(len(self.observations),1)
        self.assertEqual(actor.close_game()['status'],'FAILED_CLOSED')

    def test_wait_no_request_asymmetry_then_action(self):
        for owner in ('p1','p2'):
            actor=self.make(); self.begin(actor,owner=owner)
            for lines in (('|turn|1',),(request(owner,wait=True),)):
                self.feed(actor,lines)
                with self.assertRaises(NoAction):self.decide(actor)
                self.assertEqual(actor.status,'ACTIVE')
            self.feed(actor,(request(owner),)); self.decide(actor)
        self.assertEqual(len(self.observations),2)

    def test_canonical_switch_targets_both_seats_and_forced_switch(self):
        for owner in ('p1','p2'):
            for active_index in (0,1,2):
                actor=self.make(); self.begin(actor,owner=owner)
                self.feed(actor,(request(owner,forced=True,active_index=active_index),))
                self.action=4; snap,_=self.decide(actor)
                expected=1 if active_index!=0 else 2
                self.assertEqual(actor.choice(snap),f'switch {expected}')

    def test_raw_stable_action_mask_pp_zero_not_disabled(self):
        actor=self.make(); self.begin(actor); self.feed(actor,(request(),))
        self.decide(actor)
        mask=self.observations[-1].legal_action_mask
        self.assertTrue(mask[0]); self.assertFalse(mask[1]); self.assertTrue(mask[4]); self.assertTrue(mask[5])

    def test_stale_snapshot_and_no_repeated_action_on_same_request(self):
        actor=self.make(); self.begin(actor); self.feed(actor,(request(),))
        snap,_=self.decide(actor)
        self.feed(actor,('|turn|2',))
        with self.assertRaisesRegex(ValueError,'stale'):actor.choice(snap)
        with self.assertRaisesRegex(ValueError,'already consumed'):self.decide(actor)
        self.assertEqual(len(self.observations),1)
        self.feed(actor,(request(),)); self.decide(actor)
        self.assertEqual(len(self.observations),2)

    def test_forged_equal_snapshot_refused(self):
        actor=self.make(); self.begin(actor); self.feed(actor,(request(),))
        snap=actor.snapshot_boundary()
        with self.assertRaisesRegex(ValueError,'stale'):actor.decide(replace(snap),rng=random.Random())

    def test_wrong_owner_battle_generation_hash_and_omniscient_poison(self):
        for delta in (dict(owner='p2'),dict(owner='omniscient'),dict(battle_id='foreign'),
                      dict(generation=True),dict(generation=2),dict(prefix_sha256='0'*64)):
            actor=self.make(); self.begin(actor)
            args=dict(battle_id='authored',owner='p1',generation=0,prefix_sha256=actor.prefix_sha256,lines=(request(),))
            with self.assertRaises(ValueError):actor.accept_owner_stream(**dict(args,**delta))
            self.assertEqual(actor.status,'POISONED'); self.assertFalse(self.backends[-1].lines)

    def test_malformed_or_foreign_requests_no_partial_backend_ingest(self):
        for line in (request('p2'),'|request|{"side":{"id":"p1"},"wait":true,"wait":false}',
                     '|request|{"side":{"id":"p1"},"x":NaN}', '|request|[]'):
            actor=self.make(); self.begin(actor)
            with self.assertRaises(ValueError):self.feed(actor,('|turn|1',line))
            self.assertEqual(actor.status,'POISONED'); self.assertFalse(self.backends[-1].lines)

    def test_nonliteral_lines_and_canonical_contract_types(self):
        for lines in (['|turn|1'],(),('|turn|1\n|turn|2',),(42,),('not-wire',)):
            actor=self.make(); self.begin(actor)
            with self.assertRaises(ValueError):actor.accept_owner_stream(battle_id='authored',owner='p1',
                generation=0,prefix_sha256=actor.prefix_sha256,lines=lines)
        with self.assertRaises(ValueError):replace(self.contract,feature_masks=tuple((k,0 if k=='exact_state' else v) for k,v in self.contract.feature_masks))
        with self.assertRaises(ValueError):replace(self.contract,showdown_revision='not-source')

    def test_late_backend_failure_poison_prevents_all_decisions_and_partial_journal(self):
        actor=self.make(); self.begin(actor); self.backends[-1].fail_line='|-message|fail'
        before=actor.prefix_sha256
        with self.assertRaisesRegex(ValueError,'late line'):self.feed(actor,(request(),'|-message|fail'))
        self.assertEqual(actor.prefix_sha256,before)
        with self.assertRaises(ValueError):actor.snapshot_boundary()
        receipt=actor.close_game()
        self.assertEqual(receipt['status'],'FAILED_CLOSED'); self.assertEqual(receipt['owner_lines'],0)
        self.assertEqual(self.backends[-1].closed,1)

    def test_late_encode_or_policy_failure_is_poison_not_fallback(self):
        def explode(obs,*,rng):raise RuntimeError('fake policy failure')
        for factory in (None,lambda c:RawPolicy(c,explode,lambda:None)):
            actor=self.make(policy_factory=factory); self.begin(actor); self.feed(actor,(request(),))
            if factory is None:self.backends[-1].mutate=lambda a:a.update(numeric_features=np.zeros((23,1),np.float32))
            with self.assertRaises((ValueError,RuntimeError)):self.decide(actor)
            self.assertEqual(actor.status,'POISONED')
            with self.assertRaises(ValueError):self.decide(actor)

    def test_bad_masks_categories_nonfinite_and_illegal_policy_refuse(self):
        mutations=[lambda a:a['legal_action_mask'].__setitem__(1,True),
                   lambda a:a['categorical_ids'].__setitem__((0,0),916),
                   lambda a:a['numeric_features'].__setitem__((0,0),np.nan)]
        for mutation in mutations:
            actor=self.make(); self.begin(actor); self.feed(actor,(request(),)); self.backends[-1].mutate=mutation
            with self.assertRaises(ValueError):self.decide(actor)
            self.assertEqual(actor.status,'POISONED')
        actor=self.make(); self.begin(actor); self.feed(actor,(request(),)); self.action=1
        with self.assertRaisesRegex(ValueError,'illegal'):self.decide(actor)

    def test_terminal_close_reset_fresh_state_and_no_reused_registration(self):
        actor=self.make(); self.begin(actor); self.feed(actor,(request(),'|win|Authored'))
        with self.assertRaises(NoAction):self.decide(actor)
        receipt=actor.close_game()
        self.assertEqual(receipt['status'],'TERMINAL_UNQUALIFIED')
        self.assertFalse(receipt['public_wire_qualified']); self.assertFalse(receipt['B3_qualified'])
        with self.assertRaises(ValueError):actor.snapshot_boundary()
        self.begin(actor,battle='next',stream=2)
        self.assertEqual(actor.generation,0); self.assertEqual(actor.snapshot_boundary().lines,())
        self.assertEqual(len(self.resets),2)
        actor.close_game()
        with self.assertRaises(ValueError):self.begin(actor)

    def test_postterminal_stream_and_reused_backend_policy_alias_refused(self):
        actor=self.make(); self.begin(actor); self.feed(actor,('|tie',))
        with self.assertRaisesRegex(ValueError,'postterminal'):self.feed(actor,('|turn|2',))
        shared=FixtureBackend(self.contract)
        for _ in range(2):
            actor=self.make(backend_factory=lambda c,**kw:shared)
            if shared._claimed:
                with self.assertRaisesRegex(ValueError,'alias'):self.begin(actor)
                self.assertEqual(actor.status,'POISONED')
            else:self.begin(actor); actor.close_game()
        policy=RawPolicy(self.contract,lambda o,**k:PolicyDecision(0,'fake'),lambda:None)
        actor=self.make(policy_factory=lambda c:policy); self.begin(actor); actor.close_game()
        with self.assertRaisesRegex(ValueError,'alias'):self.begin(actor,battle='next',stream=2)

    def test_within_chunk_terminal_order_refuses_before_any_ingest(self):
        tails=[('|turn|2',),(request(),),('|win|Again',),('|tie',),
               ('|switch|p1a: Snorlax|Snorlax|100/100',),('|message|unproved annotation',)]
        for terminal in ('|win|Authored','|tie'):
            for tail in tails:
                actor=self.make(); self.begin(actor)
                with self.assertRaisesRegex(ValueError,'postterminal line'):
                    self.feed(actor,(terminal,*tail))
                self.assertEqual(self.backends[-1].lines,[])
                self.assertEqual(actor.status,'POISONED')
                with self.assertRaises(ValueError):self.decide(actor)
                receipt=actor.close_game()
                self.assertEqual(receipt['status'],'FAILED_CLOSED')
                self.assertEqual(receipt['owner_lines'],0)

    def test_within_chunk_terminal_allows_only_bare_pipe_separators(self):
        actor=self.make(); self.begin(actor)
        self.feed(actor,('|win|Authored','|','|'))
        self.assertEqual(self.backends[-1].lines,['|win|Authored','|','|'])
        with self.assertRaises(NoAction):self.decide(actor)
        self.assertEqual(actor.close_game()['status'],'TERMINAL_UNQUALIFIED')

    def test_namespace_mismatch_refuses_before_any_encode(self):
        wrong=replace(self.contract,checkpoint_sha256='e'*64)
        actor=self.make(backend_factory=lambda c,**kw:FixtureBackend(wrong))
        with self.assertRaisesRegex(ValueError,'namespace'):self.begin(actor)
        self.assertEqual(actor.status,'POISONED')

    def test_provider_namespace_drift_poisoned_and_close_failure_recorded(self):
        actor=self.make(); self.begin(actor); self.feed(actor,(request(),))
        self.backends[-1].contract=replace(self.contract,checkpoint_sha256='f'*64)
        with self.assertRaisesRegex(ValueError,'namespace drift'):self.decide(actor)
        self.assertEqual(actor.status,'POISONED')
        def fail_close():raise OSError('authored close error')
        self.backends[-1].close=fail_close
        receipt=actor.close_game()
        self.assertEqual(receipt['status'],'FAILED_CLOSED'); self.assertEqual(receipt['close_error'],'OSError')

    def test_two_backend_modes_use_identical_immutable_prefix_and_common_selection(self):
        actors=[self.make(),self.make()]
        for actor in actors:
            self.begin(actor); self.feed(actor,(request(wait=True),'|turn|1',request()))
        a,b=[actor.snapshot_boundary() for actor in actors]
        self.assertEqual(a,b)
        for actor,snapshot in zip(actors,(a,b)):
            actor.decide(snapshot,rng=random.Random(42))
        self.assertEqual(self.observations[0],self.observations[1])


class RawTransformerBindingTests(unittest.TestCase):
    def fixture(self):
        # The real policy class and binding code, without importing Torch,
        # constructing a model, accessing weights or invoking a model callback.
        from pokezero.neural_policy import TransformerSoftmaxPolicy
        c=ActorContract.champion()
        tokens=('authored',)
        payload=dict(authored='model-config')
        c=replace(c,category_vocab_sha256=hashlib.sha256(_canonical(tokens)).hexdigest(),
            model_config_sha256=hashlib.sha256(_canonical(payload)).hexdigest())
        masks=dict(c.feature_masks)
        cfg=SimpleNamespace(to_dict=lambda:payload,category_vocab=tokens,
            exact_state_enabled=masks['exact_state'],feature_pack_last_move=masks['feature_pack_last_move'],
            investment_belief_narrowing=False,item_belief_narrowing=False,stats_block_enabled=True,
            tier2_investment=True,tier2_residuals=True,transition_token_budget=0,
            observation_schema_version='pokezero.observation.v4',window_size=1,temporal_aggregator='mean',
            token_count=23,numeric_feature_count=132,categorical_feature_count=41,
            categorical_vocab_size=c.categorical_vocab_size,token_type_vocab_size=c.token_type_vocab_size)
        p=object.__new__(TransformerSoftmaxPolicy)
        p.result=SimpleNamespace(model_config=cfg); p.weights_sha256=c.checkpoint_sha256
        p.deterministic=True; p.exploration_epsilon=0.; p.sampling_temperature=1.
        p.family_gated_selection=False; p.history_mask_k=None
        p.model=SimpleNamespace(training=False,parameters=lambda:[],buffers=lambda:[])
        p.forward_fn=p._default_forward; p._history_by_player={}
        torch=SimpleNamespace(float32='authored-float32',get_float32_matmul_precision=lambda:'highest',
            backends=SimpleNamespace(mha=SimpleNamespace(get_fastpath_enabled=lambda:False),
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                cudnn=SimpleNamespace(allow_tf32=False)))
        return c,p,torch

    def test_real_policy_class_binding_and_underlying_alias_refusal(self):
        c,p,torch=self.fixture()
        with patch('pokezero.neural_policy.require_torch',return_value=torch):
            a,b=RawPolicy.from_transformer(c,p),RawPolicy.from_transformer(c,p)
            a.claim(c); a.reset()
            with self.assertRaisesRegex(ValueError,'ownership alias'):b.claim(c)

    def test_weights_remote_forward_precision_and_dtype_guards(self):
        for mode in ('weights','forward','mha','tf32','dtype','history'):
            c,p,torch=self.fixture()
            if mode=='weights':p.weights_sha256='f'*64
            if mode=='forward':p.forward_fn=lambda **kwargs:None
            if mode=='mha':torch.backends.mha.get_fastpath_enabled=lambda:True
            if mode=='tf32':torch.backends.cuda.matmul.allow_tf32=True
            if mode=='history':p.history_mask_k=0
            if mode=='dtype':
                p.model.parameters=lambda:[SimpleNamespace(device=SimpleNamespace(type='cpu'),
                    is_floating_point=lambda:True,dtype='wrong-float')]
            with patch('pokezero.neural_policy.require_torch',return_value=torch):
                with self.assertRaises(ValueError,msg=mode):RawPolicy.from_transformer(c,p)

    def test_runtime_precision_drift_refuses_before_model_callback(self):
        c,p,torch=self.fixture()
        with patch('pokezero.neural_policy.require_torch',return_value=torch):
            bound=RawPolicy.from_transformer(c,p)
        torch.backends.mha.get_fastpath_enabled=lambda:True
        with self.assertRaisesRegex(ValueError,'runtime precision'):
            bound.select_action(None,rng=random.Random())

    def test_distinct_transformers_shared_history_original_counterexample(self):
        c,p,torch=self.fixture(); _,q,_=self.fixture()
        history={'p1':['authored prior observation']}
        p._history_by_player=q._history_by_player=history
        with patch('pokezero.neural_policy.require_torch',return_value=torch):
            first,second=RawPolicy.from_transformer(c,p),RawPolicy.from_transformer(c,q)
        first.claim(c)
        with self.assertRaisesRegex(ValueError,'mutable raw history ownership alias'):second.claim(c)

    def test_distinct_history_dicts_deep_list_array_and_metadata_aliases_refuse(self):
        from pokezero.observation import PokeZeroObservationV0
        array=np.arange(4,dtype=np.float32)
        metadata={'mutable':[1]}
        obs=lambda:PokeZeroObservationV0((),(),(),(),(),metadata=metadata)
        for a,b in (([1],[1]),(array[:2],array[2:]),(obs(),obs())):
            if type(a) is list:b=a
            c,p,torch=self.fixture(); _,q,_=self.fixture()
            p._history_by_player={'p1':[a]}; q._history_by_player={'p2':[b]}
            with patch('pokezero.neural_policy.require_torch',return_value=torch):
                first,second=RawPolicy.from_transformer(c,p),RawPolicy.from_transformer(c,q)
            first.claim(c)
            with self.assertRaisesRegex(ValueError,'mutable raw history ownership alias'):second.claim(c)

    def test_immutable_model_sharing_allowed_and_owned_history_updates_leased(self):
        c,p,torch=self.fixture(); _,q,_=self.fixture()
        q.model=p.model  # Immutable model sharing is not a history alias.
        calls=[]
        def select(obs,**kwargs):
            p._history_by_player.setdefault('p1',[]).append(obs)
            calls.append(obs)
            return PolicyDecision(0,'authored')
        p.select_action=select
        with patch('pokezero.neural_policy.require_torch',return_value=torch):
            first,second=RawPolicy.from_transformer(c,p),RawPolicy.from_transformer(c,q)
        first.claim(c); second.claim(c)
        first.select_action('authored immutable observation',rng=random.Random())
        q._history_by_player['p2']=['independent']
        first.reset()
        self.assertEqual(q._history_by_player,{'p2':['independent']})
        self.assertEqual(calls,['authored immutable observation'])

    def test_shared_history_inserted_after_binding_refuses_and_history_swap_refuses(self):
        c,p,torch=self.fixture(); _,q,_=self.fixture()
        p.select_action=lambda obs,**kwargs:PolicyDecision(0,'authored')
        q.select_action=lambda obs,**kwargs:PolicyDecision(0,'authored')
        with patch('pokezero.neural_policy.require_torch',return_value=torch):
            first,second=RawPolicy.from_transformer(c,p),RawPolicy.from_transformer(c,q)
        first.claim(c); second.claim(c)
        values=[]; p._history_by_player['p1']=values
        first.select_action(None,rng=random.Random())
        q._history_by_player['p2']=values
        with self.assertRaisesRegex(ValueError,'history ownership alias'):
            second.select_action(None,rng=random.Random())
        p._history_by_player={}
        with self.assertRaisesRegex(ValueError,'history binding drift'):
            first.select_action(None,rng=random.Random())

    def test_every_bound_selection_control_function_and_namespace_drift_refuses(self):
        changes=dict(deterministic=False,exploration_epsilon=1.,sampling_temperature=2.,
            family_gated_selection=True,history_mask_k=0,device='cuda',policy_id='foreign',
            checkpoint_path='/foreign',record_policy_distribution=True,
            forward_fn=lambda t:None,select_action=lambda o,**kw:PolicyDecision(0,'changed'),
            reset=lambda:None,_default_forward=lambda **kw:None)
        for field,value in changes.items():
            c,p,torch=self.fixture(); calls=[]
            p.select_action=lambda obs,**kw:(calls.append(True),PolicyDecision(0,'original'))[1]
            with patch('pokezero.neural_policy.require_torch',return_value=torch):
                bound=RawPolicy.from_transformer(c,p)
            setattr(p,field,value)
            with self.assertRaises(ValueError,msg=field):bound.select_action(None,rng=random.Random())
            self.assertEqual(calls,[],field)
        for field,value in (('exact_state_enabled',False),('category_vocab',('foreign',))):
            c,p,torch=self.fixture()
            with patch('pokezero.neural_policy.require_torch',return_value=torch):bound=RawPolicy.from_transformer(c,p)
            setattr(p.result.model_config,field,value)
            with self.assertRaises(ValueError,msg=field):bound.select_action(None,rng=random.Random())

    def test_callable_overloaded_equality_cannot_alias_bound_function(self):
        class ForgedCallable:
            def __eq__(self, other):return True
            def __call__(self, *args, **kwargs):return PolicyDecision(0,'forged')
        for field in ('forward_fn','select_action','reset','_default_forward'):
            c,p,torch=self.fixture()
            with patch('pokezero.neural_policy.require_torch',return_value=torch):
                bound=RawPolicy.from_transformer(c,p)
            setattr(p,field,ForgedCallable())
            with self.assertRaisesRegex(ValueError,'selection/function'):
                bound.select_action(None,rng=random.Random())

    def test_late_selection_drift_refuses_after_callback(self):
        c,p,torch=self.fixture()
        def drift(obs,**kw):
            p.deterministic=False
            return PolicyDecision(0,'authored')
        p.select_action=drift
        with patch('pokezero.neural_policy.require_torch',return_value=torch):bound=RawPolicy.from_transformer(c,p)
        with self.assertRaisesRegex(ValueError,'selection/function'):
            bound.select_action(None,rng=random.Random())

    def test_history_lease_ledger_releases_dead_wrapper_without_retaining_games(self):
        import gc
        import weakref
        from pokezero.live_protocol_actor import _HISTORY_LEASES
        c,p,torch=self.fixture()
        with patch('pokezero.neural_policy.require_torch',return_value=torch):
            bound=RawPolicy.from_transformer(c,p)
        bound.claim(c)
        identity=id(p._history_by_player); reference=weakref.ref(bound)
        self.assertIn(identity,_HISTORY_LEASES)
        del bound
        gc.collect()
        self.assertIsNone(reference())
        self.assertNotIn(identity,_HISTORY_LEASES)


class ProductionPythonTests(unittest.TestCase):
    def test_real_incremental_python_both_owners_own_request_interleaving(self):
        from test_saved_player_projection import ProductionHistoryTests, leads, request as own_request
        f=ProductionHistoryTests(); f.setUp(); self.addCleanup(f.doCleanups)
        f.source.metadata=SimpleNamespace(source_hash='b'*16)
        for owner in ('p1','p2'):
            contract=replace(ActorContract.champion(),belief_set_source_hash='b'*16,
                category_vocab_sha256=_vocab_sha(f.config.category_vocab),
                categorical_vocab_size=f.config.category_vocab.size)
            backends=[]; observations=[]
            def backend(c,**kw):
                b=PythonBackend(c,config=f.config,**kw); backends.append(b); return b
            def select(obs,*,rng):observations.append(obs); return PolicyDecision(0,'authored')
            actor=LiveProtocolActor(contract,backend_factory=backend,
                policy_factory=lambda c:RawPolicy(c,select,lambda:None))
            actor.begin_game(battle_id='authored',owner=owner,seed=7,stream_id=1)
            lines=list(leads())
            peer='p2' if owner=='p1' else 'p1'
            for i,line in enumerate(lines):
                if line.startswith('|switch|'+peer):lines[i]=line.rsplit('|',1)[0]+'|75/100'
            lines += [own_request(owner,wait=True),'|-message|authored actual-owner hint',
                      own_request(owner,move='thunderbolt')]
            actor.accept_owner_stream(battle_id='authored',owner=owner,generation=0,
                prefix_sha256=actor.prefix_sha256,lines=tuple(lines))
            snap=actor.snapshot_boundary(); actor.decide(snap,rng=random.Random(7))
            self.assertIs(actor.decision_observation(snap),observations[0])
            self.assertEqual(backends[0].env.protocol_lines,tuple(lines))
            self.assertEqual(len(backends[0].env._request_history[owner]),2)
            self.assertEqual(set(backends[0].env._parser.requests),{owner})
            self.assertEqual(observations[0].perspective.player_id,owner)
            self.assertEqual(actor.close_game()['status'],'ABORTED_UNQUALIFIED')

    def test_real_python_refuses_within_chunk_terminal_before_mutating_history(self):
        from test_saved_player_projection import ProductionHistoryTests, leads, request as own_request
        f=ProductionHistoryTests(); f.setUp(); self.addCleanup(f.doCleanups)
        f.source.metadata=SimpleNamespace(source_hash='b'*16)
        contract=replace(ActorContract.champion(),belief_set_source_hash='b'*16,
            category_vocab_sha256=_vocab_sha(f.config.category_vocab),
            categorical_vocab_size=f.config.category_vocab.size)
        for tail in (('|turn|2',),(own_request('p1'),)):
            backend=PythonBackend(contract,config=f.config,battle_id='terminal',owner='p1',stream_id=1)
            actor=LiveProtocolActor(contract,backend_factory=lambda c,**kw:backend,
                policy_factory=lambda c:RawPolicy(c,lambda o,**kw:PolicyDecision(0,'fake'),lambda:None))
            actor.begin_game(battle_id='terminal',owner='p1',seed=42,stream_id=1)
            with self.assertRaisesRegex(ValueError,'postterminal line'):
                actor.accept_owner_stream(battle_id='terminal',owner='p1',generation=0,
                    prefix_sha256=actor.prefix_sha256,lines=tuple(leads())+('|win|Authored',*tail))
            self.assertEqual(backend.env.protocol_lines,())
            self.assertEqual(actor.close_game()['status'],'FAILED_CLOSED')


class ProductionJaxTests(unittest.TestCase):
    def test_real_jax_adapter_catalog_namespace_and_action_targets(self):
        import jax
        from test_gpu_engine_v4_encoder import V4EncoderTests
        V4EncoderTests.setUpClass(); f=V4EncoderTests()
        contract=ActorContract('a'*64,_vocab_sha(f.catalog.vocabulary),f.catalog.source.source_hash,
            'd'*64,tuple((k,f.masks[k]) for k in MASK_NAMES),'e'*40,
            categorical_vocab_size=f.catalog.vocabulary.size)
        with (patch('subprocess.Popen',side_effect=AssertionError('no simulator')),
              patch('subprocess.run',side_effect=AssertionError('no runtime')),
              jax.default_matmul_precision('highest'), jax.enable_x64(False)):
            observations=[]; backends=[]
            def backend(c,**kwargs):
                b=JaxBackend(c,catalog=f.catalog,**kwargs); backends.append(b); return b
            def select(obs,*,rng):
                observations.append(obs); return PolicyDecision(4,'authored-jax')
            actor=LiveProtocolActor(contract,backend_factory=backend,
                policy_factory=lambda c:RawPolicy(c,select,lambda:None))
            actor.begin_game(battle_id='jax-owning',owner='p1',seed=3,stream_id=1)
            own='|request|'+json.dumps(f.request(0))
            wait=json.loads(own[len('|request|'):]); wait.pop('active'); wait['wait']=True
            lines=tuple(f.public.prefix)+('|request|'+json.dumps(wait),'|turn|1',own)
            actor.accept_owner_stream(battle_id=actor.battle_id,owner=actor.owner,
                generation=0,prefix_sha256=actor.prefix_sha256,lines=lines)
            snap=actor.snapshot_boundary(); actor.decide(snap,rng=random.Random(3))
            self.assertIs(actor.decision_observation(snap),observations[0])
            self.assertEqual(actor.choice(snap),'switch 1')
            self.assertEqual(observations[0].schema_version,'pokezero.observation.v4')
            self.assertEqual(snap.lines,lines); self.assertEqual(snap.request_count,2)
            self.assertEqual(backends[0].adapter.battle_id,'jax-owning')
            actor.close_game()
            wrong=replace(contract,category_vocab_sha256='f'*64)
            with self.assertRaisesRegex(ValueError,'contract differs'):
                JaxBackend(wrong,catalog=f.catalog,battle_id='bad',owner='p1',stream_id=2)
