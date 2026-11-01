"""Pinned owner-only entry callback/component fixtures, no battle/kernel/model."""
from copy import deepcopy
from pathlib import Path
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import entry, engine_pressure_callbacks as pressure
from gpu_engine import engine_lifecycle_callbacks as life, engine_public_events as events
from gpu_engine import engine_public_event_device as base, engine_full_v4_event_device as full
from gpu_engine import v4_encoder as v4, public_wire, public_inputs as inputs
from gpu_engine.state import empty_state
import test_gpu_engine_engine_public_events as engine_fixture
import test_gpu_engine_v4_encoder as encoder_fixture


class PressureEntryOwningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Static literal catalog/compiler fixtures only: neither constructor nor
        # transition.step/initialize is called by this suite.
        engine_fixture.EnginePublicEventsTests.setUpClass()
        encoder_fixture.V4EncoderTests.setUpClass()
        f=engine_fixture.EnginePublicEventsTests
        cls.catalog=encoder_fixture.V4EncoderTests.catalog
        cls.metadata=deepcopy(f.metadata)
        cls.metadata['vocabulary'].update(species=['_none','snorlax','absol','gardevoir'],
            abilities=['_none','pressure','trace'],items=['_none'],types=['_none','normal','fighting','flying','poison','ground',
                'rock','bug','ghost','steel','fire','water','grass','electric','psychic','ice','dragon','dark'])
        cls.entry=entry.from_metadata(cls.metadata)
        cls.engine=f.engine_table._replace(entry=cls.entry)
        cls.producer=events.compile_catalog(cls.metadata,cls.engine,public_moves=cls.catalog.moves,
            dex=encoder_fixture.V4EncoderTests.inputdex,exposure=f.exposure)
        cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,
            expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
        cls.projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,
            expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
        cls.compiled=full.compile_adapter(cls.metadata,cls.projector,catalog=cls.catalog)
        cls.fast=staticmethod(jax.jit(lambda stream,tape,before,after:
            full.project_device(stream,tape,before,after,cls.compiled.device)))

    def state(self,b=2):
        s=empty_state(b,seed=29)
        return s._replace(species=j.ones_like(s.species).at[:,1].set(2),
            level=j.full_like(s.level,80),gender=j.ones_like(s.gender),hp=j.full_like(s.hp,100),
            max_hp=j.full_like(s.hp,100),ability=j.zeros_like(s.ability).at[:,1,5].set(1),
            base_ability=j.zeros_like(s.base_ability).at[:,1,5].set(1),
            active=j.broadcast_to(j.array([3,5],j.int32),(b,2)),types=s.types.at[...,0].set(1))

    def tape(self,state=None):
        state=self.state() if state is None else state;b=state.hp.shape[0]
        cursor=life.register(j.arange(101,101+b,dtype=j.int32),**self.kw)
        before=life.snapshot(cursor);tape=life._tape(cursor,self.producer,64)
        for side in (0,1):
            tape=life._reveal(tape,state,j.full(b,side,j.int32),j.full(b,-1,j.int32),state.active[:,side],
                eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=True)
        return state,tape,before

    @staticmethod
    def words(p,row):
        return np.asarray(p.frames.words).transpose(1,0,2,3)[row][np.asarray(p.frames.valid).transpose(1,0,2)[row]]

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def test_effective_source_owner_split_and_original_writer_words_both_owners(self):
        root=Path('/Users/scott/workspace/agents/pokemon-showdown-timing-smoke-f76228a')
        source=(root/'data/mods/gen3/abilities.ts').read_text()
        body=source.split('\tpressure: {',1)[1].split('\n\t},',1)[0]
        self.assertIn("this.addSplit(pokemon.side.id, ['-ability', pokemon, 'Pressure', '[silent]']);",body)
        s,t,before=self.tape()
        t=pressure.entry_pressure(t,1,s.active[:,1],s.ability[j.arange(2),1,s.active[:,1]],eligible=j.ones(2,bool))
        t=t._replace(cursor=t.cursor._replace(initialized=j.ones(2,bool)))
        t=life._turn(t,j.ones(2,bool));after=life.snapshot(t.cursor)
        stream=full.register(t.stream_id,j.array([0,1],j.int32),compiled=self.compiled)
        p=self.fast(stream,t,before,after)
        self.assertEqual(p.error.tolist(),[0,0])
        for row,owner in enumerate((0,1)):
            lines=['|switch|p1a: Snorlax|Snorlax, L80, M|100/100',
                   '|switch|p2a: Absol|Absol, L80, M|100/100']
            if owner==1:lines+=['|-ability|p2a: Absol|Pressure|[silent]']
            lines+=['|turn|1']
            _,ps=public_wire.project_lines(public_wire.WireState(),lines,**v4.wire_kwargs(self.catalog))
            expected=np.array([e.words() for projection in ps for e in projection.events],np.int32)
            np.testing.assert_array_equal(self.words(p,row),expected)
        self.assertEqual(p.state.causal.sequence.tolist(),[4,4])
        self.assertNotEqual(*p.state.sequence.tolist())
        for flag in (p.full_line_consumers_complete,p.full_encoder_admitted,p.B4_qualified):self.assertFalse(bool(flag.any()))

    def test_actual_entry_component_pretrace_noop_dead_inactive_and_numeric_identity(self):
        s,t,_=self.tape(self.state(4));s=s._replace(
            ability=s.ability.at[1,1,5].set(2).at[1,0,3].set(1),
            base_ability=s.base_ability.at[1,1,5].set(2).at[1,0,3].set(1),
            hp=s.hp.at[2,1,5].set(0))
        mask=j.array([True,True,True,False])
        plain,ko=entry.enter(s,1,self.entry,eligible=mask,hazards=False)
        traced,kt,out=entry.enter(s,1,self.entry,eligible=mask,hazards=False,_weather_trace=t)
        self.same(plain,traced);self.same(ko,kt)
        self.assertEqual(out.count.tolist(),[3,2,2,2])
        self.assertEqual(int(out.words[0,2,0]),pressure.PRESSURE_START)
        self.assertEqual(int(out.words[0,2,4]),5)
        self.assertEqual(int(traced.ability[1,1,5]),1) # Trace gains Pressure, no onStart.
        self.assertTrue(bool(out.refusal[1]&256))
        self.assertEqual(out.words[3].tolist(),t.words[3].tolist())

    def test_bad_identity_generation_and_peer_invisible_receipt_refuse_whole_row(self):
        s,t,before=self.tape();t=pressure.entry_pressure(t,1,s.active[:,1],j.ones(2,j.int32),eligible=j.ones(2,bool))
        t=t._replace(cursor=t.cursor._replace(initialized=j.ones(2,bool)))
        t=life._turn(t,j.ones(2,bool));after=life.snapshot(t.cursor)
        stream=full.register(t.stream_id,j.array([0,1],j.int32),compiled=self.compiled)
        for name,bad in (
            ('ability',t._replace(words=t.words.at[0,2,5].set(2))),
            ('physical',t._replace(words=t.words.at[0,2,4].set(4))),
            ('generation',t._replace(event_generation=t.event_generation.at[0,2,1].set(1))),
            ('reserved',t._replace(words=t.words.at[0,2,8].set(1)))):
            with self.subTest(name=name):
                p=self.fast(stream,bad,before,after)
                self.assertNotEqual(int(p.error[0]),0);self.assertEqual(int(p.error[1]),0)
                self.assertEqual(len(self.words(p,0)),0)
                self.assertEqual(int(p.state.sequence[0]),0)
                self.assertEqual(int(p.state.causal.sequence[0]),0)

    def test_helper_b8_inactive_overflow_priorflags_and_legacy_noop(self):
        s,t,_=self.tape(self.state(8))
        t=t._replace(refusal=t.refusal.at[1].set(t.refusal[1]|256),
            count=t.count.at[2].set(64))
        mask=j.array([True,True,True,False,False,False,False,False])
        fast=jax.jit(lambda tape:pressure.entry_pressure(tape,1,s.active[:,1],j.ones(8,j.int32),eligible=mask))
        out=fast(t)
        self.assertEqual(out.count.tolist(),[3,3,64,2,2,2,2,2])
        self.assertTrue(bool(out.refusal[1]&256));self.assertTrue(bool(out.refusal[2]&events.OVERFLOW))
        for x,y in zip(jax.tree.leaves(out),jax.tree.leaves(t)):
            if getattr(x,'ndim',0) and x.shape[0]==8:np.testing.assert_array_equal(x[3:],y[3:])
        legacy=events.initialize(8,64,catalog=self.producer)
        self.assertIs(pressure.entry_pressure(legacy,1,s.active[:,1],j.ones(8,j.int32),eligible=mask),legacy)
