"""Literal actual-core status writes into existing FULL consumers, no games."""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np
import jax
import jax.numpy as j

from pokezero.dex import ShowdownDex
from pokezero.category_vocab import CategoryVocabulary
from gpu_engine import transition as t, engine_lifecycle_callbacks as life, engine_public_events as events
from gpu_engine import engine_public_event_device as base, engine_full_v4_event_device as full
from gpu_engine import engine_status_callbacks as status, public_wire, public_inputs as inputs, v4_encoder as v4
import test_gpu_engine_engine_own_facts as fixture


class StatusCallbacksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.DeviceOwnFactsTests.setUpClass(); cls.f=fixture.DeviceOwnFactsTests
        cls.metadata=deepcopy(cls.f.metadata)
        cls.metadata['vocabulary']['items'].append('lumberry')
        cls.metadata['vocabulary']['abilities'].append('synchronize')
        moves=dict(cls.f.dex.moves)
        cls.ids={}
        for name,typ,s,pp in [('thunderwave','Electric','par',20),('sleeppowder','Grass','slp',15)]:
            cls.ids[name]=len(cls.metadata['moves'])
            cls.metadata['vocabulary']['moves'].append(name)
            cls.metadata['moves'].append(dict(id=name,base_power=0,
                type=cls.metadata['vocabulary']['types'].index(typ.lower()),source_type=typ,
                category='Status',pp=pp,accuracy=100,priority=0,flags={}))
            moves[name]=replace(cls.f.dex.moves['surf'],id=name,name=name,type=typ,category='Status',
                gen3_category='Status',base_power=0,pp=pp,target='normal',status=s)
        dex=ShowdownDex(moves,cls.f.dex.species,cls.f.dex.type_chart)
        labels=set(cls.f.catalog.vocabulary.tokens)
        for name in cls.ids: labels|={'move:'+name,'belief:possible_move:'+name}
        vocab=CategoryVocabulary(tuple(sorted(labels)))
        cls.catalog=v4.compile_encoder(cls.f.host,dex,vocab,checkpoint_sha256='a'*64,
            feature_masks=cls.f.catalog.masks.__dict__)
        cls.table=t.tables_from_metadata(cls.metadata)
        cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=dex,
            exposure=events.HealthExposure(events.SHOWDOWN,'gen3randombattle',True,False))
        p=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,
            expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
        cls.compiled=full.compile_adapter(cls.metadata,p,catalog=cls.catalog)
        cls.fast=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.compiled.device,p,cls.catalog)))

    def kwargs(self):return dict(catalog=self.producer,expected_binding=self.producer.binding,
        expected_engine_binding=np.array(self.producer.engine_binding,np.uint32))

    def initial(self,name='thunderwave',arrays=None,status_value=0):
        arrays={k:np.array(v,copy=True) for k,v in (self.f.arrays if arrays is None else arrays).items()}
        arrays['moves'][0,0,:2]=[self.ids[name],1]
        state=t.state_from_pool(arrays,np.array([[0,1],[0,1]],np.int64),seed=31,table=None)
        state=state._replace(gender=j.array(np.stack([arrays['gender']]*2)),
            status=state.status.at[:,1,0].set(status_value))
        ids=j.array([71,72],j.int32)
        return life.initialize_with_lifecycle(state,self.table,cursor=life.register(ids,**self.kwargs()),
            tie_coin=j.zeros(2,j.int32),**self.kwargs())

    def chance(self):
        draws=j.zeros((2,2,12),j.int32).at[:,:,4].set(1)
        return t.ChanceInputs(draws,*[j.zeros((2,2),j.int32) for _ in range(4)],j.zeros(2,j.int32))

    def step(self,frame,actions):
        a=j.array(actions,j.int32)
        traced=life.trace_step(frame.state,a,self.table,cursor=frame.cursor,chance_inputs=self.chance(),**self.kwargs())
        legacy=t.step_with_chance(frame.state,a,self.table,self.chance())
        self.same(traced.state,legacy.state)
        # Legacy synthetic public tape receives no new lifecycle/status kinds.
        old=events.trace_step(frame.state,a,self.table,chance_inputs=self.chance(),**self.kwargs())
        self.same(old.step.state,legacy.state)
        self.assertTrue(bool(j.all((old.callbacks.words[:,:,0]>=0)&(old.callbacks.words[:,:,0]<=3))))
        return traced

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def start(self,frame,owners):
        return full.register(frame.cursor.stream_id,owners,compiled=self.compiled),v4.initialize(
            self.catalog,stream_ids=frame.cursor.stream_id,owner_sides=owners)

    def compare(self,state,frame,current,wire):
        lower=full.causal.project_frame(state.causal,frame,self.compiled.causal.device)
        self.assertEqual(lower.error.tolist(),[0,0])
        rows=[]; next_wire=[]; tokens=('', 'slp','par','brn','frz','psn','tox')
        status_map=[0,2,3,5,6,1,4]
        for row in range(2):
            lines=[]; active=dict(wire[row].active_idents())
            words=np.array(lower.chunk.words[row])[np.array(lower.chunk.valid[row])]
            for index,w in enumerate(words):
                k,_,turn,_,actor,_,target,_,*p=map(int,w)
                if k==inputs.Kind.SWITCH:
                    name=self.catalog.species.names[p[1]]; active[target]=f'p{target+1}a: {name}'
                    lines.append(f'|switch|{active[target]}|{name}, L{p[3]}'+
                        (', M' if p[4]==1 else ', F' if p[4]==2 else '')+f'|{p[5]}/{p[6]}'+(' '+tokens[p[7]] if p[7] else ''))
                elif k==inputs.Kind.MOVE:lines.append(f'|move|{active[actor]}|{self.catalog.moves.names[p[1]]}|{active[target]}')
                elif k==inputs.Kind.CONDITION:lines.append(f'|-damage|{active[target]}|'+
                    ('0 fnt' if p[4] else f'{p[1]}/{p[2]}'+(' '+tokens[p[3]] if p[3] else '')))
                elif k==inputs.Kind.STATUS:
                    old=status_map[int(frame.callbacks.words[row,index,5])]
                    # Pinned Gen3 sleep Start names the source move. This
                    # non-Rest tag must have exactly the compiled plain STATUS
                    # control words; Rest remains explicitly UNKNOWN.
                    lines.append(f'|-status|{active[target]}|{tokens[p[1]]}'+
                        ('|[from] move: Sleep Powder' if p[1]==1 else '') if p[1] else
                        f'|-curestatus|{active[target]}|{tokens[old]}|[msg]')
                elif k==inputs.Kind.ITEM:
                    old=status_map[int(frame.callbacks.words[row,index,6])]
                    lines.append(f'|-enditem|{active[target]}|Lum Berry|[eat]')
                    if old:lines.append(f'|-curestatus|{active[target]}|{tokens[old]}|[msg]')
                    if int(frame.callbacks.words[row,index,8]):lines.append(f'|-end|{active[target]}|confusion')
                elif k==inputs.Kind.FIELD:
                    proc=int(frame.callbacks.words[row,index,7]);source=int(frame.callbacks.words[row,index,1])
                    label=('', 'RainDance','SunnyDay','Sandstorm')[proc]
                    ability=('', 'Drizzle','Drought','Sand Stream')[proc]
                    lines.append(f'|-weather|{label}|[from] ability: {ability}|[of] {active[source]}')
                elif k==inputs.Kind.TURN:lines.append(f'|turn|{turn}')
                elif k==inputs.Kind.ACTION_CHUNK_END:lines.append('|')
                elif k==inputs.Kind.UPKEEP:lines.append('|upkeep')
                elif k==inputs.Kind.FAINT:lines.append('|faint|'+active[target])
                else:self.fail('unexpected semantic kind')
            new,ps=public_wire.project_lines(wire[row],lines,**v4.wire_kwargs(self.catalog))
            rows.append([e for p in ps for e in p.events]);next_wire.append(new)
        ns={k:np.array(getattr(self.compiled.device.namespaces,k)) for k in full._NAMES if k!='source_binding'}
        expected=list(inputs.pack_chunks(np.array(frame.cursor.stream_id),rows,
            source_binding=np.array(self.catalog.source.device.binding),**ns))
        want=current
        for chunk in expected:
            want,error=v4.fold_chunk(want,chunk,self.catalog);self.assertEqual(error.tolist(),[0,0])
        result=self.fast(state,frame,current);self.assertEqual(result.projection.error.tolist(),[0,0])
        self.same(result.projection.chunk,expected[0]);self.same(result.public,want)
        for i,chunk in enumerate(expected):
            self.same(result.projection.frames.words[i],chunk.words)
            self.same(result.projection.frames.valid[i],chunk.valid)
            self.same(result.projection.frames.more[i],chunk.more)
        self.assertFalse(bool(result.projection.frames.valid[len(expected):].any()))
        self.assertFalse(bool(result.projection.full_encoder_admitted.any()))
        self.assertFalse(bool(result.projection.B4_qualified.any()))
        return result,next_wire

    def test_actual_primary_status_exact_full_words_both_observers_and_legacy_rng(self):
        frame=self.initial()
        for side in (0,1):
            state,current=self.start(frame,j.full(2,side,j.int32))
            result,wire=self.compare(state,frame,current,[public_wire.WireState()]*2)
            actual=self.step(frame,[[0,0],[0,0]])
            self.assertEqual(actual.callbacks.words[0,:int(actual.callbacks.count[0]),0].tolist(),[1,9,7,5,4])
            self.assertEqual(actual.callbacks.words[0,1].tolist(),[9,-1,-1,1,0,0,1,0,0,0])
            self.assertEqual(actual.callbacks.refusal.tolist(),[events.UNINSTRUMENTED]*2)
            result,_=self.compare(result.projection.state,actual,result.public,wire)
            self.assertEqual(result.public.mons.status[:,1,0].tolist(),[2,2])
            self.assertEqual(result.public.context.sleep.status[:,1,0].tolist(),[2,2])

    def test_hot_jit_same_actual_core_callback_order_and_fresh_static_guards(self):
        frame=self.initial();actions=j.zeros((2,2),j.int32)
        eager=self.step(frame,actions)
        tape=life._tape(frame.cursor,self.producer,64)
        fast=jax.jit(lambda s,a,c,tape:t.step_with_chance(s,a,self.table,c,
            _public_trace=tape,_lifecycle_trace=True))
        for _ in range(2):
            events.check_catalog(self.producer,self.table,expected_binding=self.producer.binding,
                expected_engine_binding=np.array(self.producer.engine_binding,np.uint32))
            actual=fast(frame.state,actions,self.chance(),tape)
            self.same(actual.step.state,eager.state);self.same(actual.callbacks,eager.callbacks)
        self.assertEqual(fast._cache_size(),1)
        original=self.compiled.device
        object.__setattr__(self.compiled,'device',original._replace(statuses=np.array(original.statuses)))
        try:
            with self.assertRaises(ValueError):full.check_adapter(self.compiled)
        finally:object.__setattr__(self.compiled,'device',original)
        closed=jax.make_jaxpr(fast)(frame.state,actions,self.chance(),tape)
        def primitives(program):
            names=set()
            for equation in program.jaxpr.eqns:
                names.add(equation.primitive.name)
                for value in equation.params.values():
                    for child in value if isinstance(value,(tuple,list)) else (value,):
                        if hasattr(child,'jaxpr'):names.update(primitives(child))
            return names
        self.assertFalse(primitives(closed)&{'pure_callback','io_callback','host_callback'})

    def test_actual_sleep_then_wake_has_old_status_hp_before_cure_then_move(self):
        frame=self.initial('sleeppowder')
        state,current=self.start(frame,j.array([0,1],j.int32))
        result,wire=self.compare(state,frame,current,[public_wire.WireState()]*2)
        first=self.step(frame,[[0,0],[0,0]])
        result,wire=self.compare(result.projection.state,first,result.public,wire)
        self.assertEqual(result.public.mons.status[:,1,0].tolist(),[1,1])
        second=self.step(first,[[1,0],[1,0]])
        kinds=second.callbacks.words[0,:int(second.callbacks.count[0]),0].tolist()
        self.assertIn(status.STATUS_CURED,kinds)
        cure=kinds.index(status.STATUS_CURED)
        self.assertEqual(second.callbacks.words[0,cure,5:8].tolist(),[5,0,status.WAKE])
        prior_hp=kinds.index(events.MAIN_HP)
        self.assertLess(prior_hp,cure)
        self.assertEqual(second.callbacks.words[0,prior_hp,7].item(),5)
        self.assertEqual(kinds[cure+1],events.MOVE)
        result,_=self.compare(result.projection.state,second,result.public,wire)
        self.assertEqual(result.public.mons.status[:,1,0].tolist(),[0,0])

    def test_valid_enum_wrong_old_status_cure_rolls_back_only_affected_row(self):
        frame=self.initial('sleeppowder');state,p=self.start(frame,j.array([0,1],j.int32))
        initial=self.fast(state,frame,p)
        first=self.step(frame,[[0,0],[0,0]])
        asleep=self.fast(initial.projection.state,first,initial.public)
        second=self.step(first,[[1,0],[1,0]])
        kinds=np.array(second.callbacks.words[0,:,0]);index=int(np.flatnonzero(kinds==status.STATUS_CURED)[0])
        wrong=second.callbacks.words.at[0,index,5].set(6).at[0,index,7].set(status.THAW)
        tampered=second._replace(callbacks=second.callbacks._replace(words=wrong))
        # This deliberately passes the enum/reason grammar; only prior causal
        # public status distinguishes the false freeze cure from actual wake.
        lower=full.causal.project_frame(asleep.projection.state.causal,tampered,self.compiled.causal.device)
        self.assertEqual(lower.error.tolist(),[0,0])
        result=self.fast(asleep.projection.state,tampered,asleep.public)
        self.assertEqual(result.projection.error.tolist(),[int(inputs.Error.IDENTITY),0])
        self.assertFalse(bool(result.projection.chunk.valid[0].any()))
        self.assertTrue(bool(result.projection.chunk.valid[1].any()))
        self.assertEqual(result.public.cursor.sequence[0].item(),asleep.public.cursor.sequence[0].item())
        self.assertEqual(result.public.mons.status[:,1,0].tolist(),[1,0])

    def test_unproved_lum_and_nonzero_entry_prefix_refuse_entire_rows(self):
        arrays={k:np.array(v,copy=True) for k,v in self.f.arrays.items()}
        arrays['item'][1,0]=self.metadata['vocabulary']['items'].index('lumberry')
        frame=self.initial(arrays=arrays)
        actual=self.step(frame,[[0,0],[0,0]])
        self.assertFalse(bool((actual.callbacks.refusal & status.UNKNOWN_STATUS).any()))
        self.assertEqual(actual.state.status[:,1,0].tolist(),[0,0])
        state,current=self.start(frame,j.array([0,1],j.int32))
        initial=self.fast(state,frame,current)
        refused=self.fast(initial.projection.state,actual,initial.public)
        self.assertFalse(bool(refused.projection.chunk.valid.any()))
        self.same(refused.public,initial.public)
        unsupported=self.initial(status_value=3)  # real Immunity entry cure
        self.assertTrue(bool((unsupported.callbacks.refusal & status.UNKNOWN_STATUS).all()))
        registered,public=self.start(unsupported,j.array([0,1],j.int32))
        result=self.fast(registered,unsupported,public)
        self.assertFalse(bool(result.projection.chunk.words.any()))

    def test_private_counterfactuals_and_mixed_unknown_do_not_supply_public_facts(self):
        baseline=self.initial()
        baseline_step=self.step(baseline,[[0,0],[0,0]])
        for owner in (0,1):
            arrays={k:np.array(v,copy=True) for k,v in self.f.arrays.items()}; peer=1-owner
            arrays['stats'][peer,:,0]+=37
            arrays['stats'][peer,:,1:5]+=23  # preserve speed and causal order
            arrays['item'][peer]=0
            arrays['ability'][peer]=np.roll(arrays['ability'][peer],1)
            arrays['hidden_power_type'][peer]=9;arrays['hidden_power_power'][peer]=31
            changed=self.initial(arrays=arrays)
            changed_step=self.step(changed,[[0,0],[0,0]])
            owners=j.full(2,owner,j.int32)
            a,p=self.start(baseline,owners);b,q=self.start(changed,owners)
            left=self.fast(a,baseline,p);right=self.fast(b,changed,q)
            self.same(left.projection.chunk,right.projection.chunk);self.same(left.public,right.public)
            left=self.fast(left.projection.state,baseline_step,left.public)
            right=self.fast(right.projection.state,changed_step,right.public)
            self.assertEqual(left.projection.error.tolist(),[0,0]);self.assertEqual(right.projection.error.tolist(),[0,0])
            self.same(left.projection.chunk,right.projection.chunk);self.same(left.public,right.public)
        arrays={k:np.array(v,copy=True) for k,v in self.f.arrays.items()}
        arrays['item'][1,0]=self.metadata['vocabulary']['items'].index('lumberry')
        frame=self.initial(arrays=arrays)
        state,p=self.start(frame,j.array([0,1],j.int32));initial=self.fast(state,frame,p)
        mixed=self.step(frame,[[0,0],[1,0]])
        self.assertEqual((mixed.callbacks.refusal & status.UNKNOWN_STATUS).tolist(),[0,0])
        result=self.fast(initial.projection.state,mixed,initial.public)
        self.assertNotEqual(result.projection.error[0].item(),0)
        self.assertEqual(result.projection.error[1].item(),0)
        self.assertFalse(bool(result.projection.chunk.valid[0].any()))
        self.assertTrue(bool(result.projection.chunk.valid[1].any()))
        self.assertEqual(result.public.cursor.sequence[0].item(),initial.public.cursor.sequence[0].item())
        self.assertEqual(self.fast._cache_size(),1)

    def test_actual_nested_reflection_is_unknown_before_private_ability_attribution(self):
        arrays={k:np.array(v,copy=True) for k,v in self.f.arrays.items()}
        arrays['ability'][1,0]=self.metadata['vocabulary']['abilities'].index('synchronize')
        frame=self.initial(arrays=arrays)
        state,p=self.start(frame,j.array([0,1],j.int32));initial=self.fast(state,frame,p)
        actual=self.step(frame,[[0,0],[0,0]])
        self.assertEqual(actual.state.status[:,:,0].tolist(),[[1,1],[1,1]])
        self.assertTrue(bool((actual.callbacks.refusal & status.UNKNOWN_STATUS).all()))
        result=self.fast(initial.projection.state,actual,initial.public)
        self.assertFalse(bool(result.projection.chunk.words.any()))
        self.same(result.public,initial.public)


if __name__=='__main__':unittest.main()
