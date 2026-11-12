"""Fixture for literal actual-core status writes into existing FULL consumers, no games (not a test module).

StatusCallbacksTests keeps the fixture API of the former test_gpu_engine_engine_status_callbacks:
setUpClass, kwargs, initial, chance, step, same, start and compare. Its tests run in
test_gpu_engine_status_callbacks_public and test_gpu_engine_status_callbacks_hot_jit. The borrowers are
engine_lum_public, engine_weather_public and those two modules. gpu_engine_shared_public_jits.compare
swaps this module's own full and v4 names during a compare call, so those imports stay.
"""
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
from gpu_engine import public_wire, public_inputs as inputs, v4_encoder as v4
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
                elif k==inputs.Kind.MOVE:
                    still=index+1<int(frame.callbacks.count[row]) and int(frame.callbacks.words[row,index+1,0])==15
                    lines.append(f'|move|{active[actor]}|{self.catalog.moves.names[p[1]]}|{active[target]}'+('|[still]' if still else ''))
                elif k==inputs.Kind.ACTIVE:
                    self.assertEqual(int(frame.callbacks.words[row,index,0]),15)
                    lines.append(f'|-fail|{active[target]}')
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
                elif k==inputs.Kind.ABILITY:
                    # Literal Gen3 Pressure onStart is visible only to its
                    # owning seat. Keep the original raw-index alignment.
                    self.assertEqual(int(frame.callbacks.words[row,index,0]),24)
                    if int(state.causal.owner[row])==target:
                        lines.append(f'|-ability|{active[target]}|Pressure|[silent]')
                elif k==inputs.Kind.FIELD:
                    raw=frame.callbacks.words[row,index];label=('none','RainDance','SunnyDay','Sandstorm','Hail')[int(raw[6])]
                    if int(raw[0])==12:
                        proc=int(raw[7]);source=int(raw[1]);ability=('', 'Drizzle','Drought','Sand Stream')[proc]
                        lines.append(f'|-weather|{label}|[from] ability: {ability}|[of] {active[source]}')
                    else:lines.append(f'|-weather|{label}'+('|[upkeep]' if int(raw[0])==14 and int(raw[9])==1 else ''))
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
