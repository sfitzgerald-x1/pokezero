"""Tiny actual selected-kernel writes and original both-owner FULL consumers.

The explicit component prefix supplies public pre-state; it does not attest
constructor/source history or complete B4/physics coverage.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import transition,entry,engine_public_events as events
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_event_device as base
from gpu_engine import engine_full_v4_event_device as full,engine_execution_completeness as execution
from gpu_engine import engine_substitute_callbacks as sub,public_wire,v4_encoder as v4
from gpu_engine.state import empty_state
from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex
import test_gpu_engine_v4_encoder as encoder_fixture

class SubstituteOwningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        encoder_fixture.V4EncoderTests.setUpClass();f=encoder_fixture.V4EncoderTests
        # Literal selected canonical move metadata; original finite parser and
        # source catalog remain unchanged. No Battle/native/roster is created.
        original=f.inputdex.moves['protect']
        move=replace(original,id='substitute',name='Substitute',base_power=0,type='Normal',
            category='Status',target='self',priority=0,effect_label='substitute',effect_chance=100,self_hp_cost=.25,pp=10)
        dex=ShowdownDex(dict(f.inputdex.moves,substitute=move),dict(f.inputdex.species),f.inputdex.type_chart)
        labels=set(f.vocab.tokens)|{'move:substitute','belief:possible_move:substitute','move_category:Status','move_priority:0','move_effect:substitute'}
        cls.catalog=v4.compile_encoder(f.source,dex,CategoryVocabulary(tuple(sorted(labels))),
            checkpoint_sha256='a'*64,feature_masks=f.masks)
        types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost','steel','fire','water','grass','electric','psychic','ice','dragon','dark']
        cls.metadata=dict(vocabulary=dict(species=['_none','snorlax','absol','gardevoir'],
            abilities=['_none'],items=['_none'],types=types,moves=['_none','substitute','struggle']),
            moves=[None,dict(id='substitute',base_power=0,type=1,source_type='Normal',category='Status',accuracy=100,pp=10,priority=0,flags={}),
                dict(id='struggle',base_power=50,type=1,source_type='Normal',category='Physical',accuracy=100,pp=10,priority=0,flags={})],
            type_chart=[[1]*17 for _ in range(17)])
        cls.table=transition.tables_from_metadata(cls.metadata)
        cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=dex,
            exposure=events.HealthExposure(events.SHOWDOWN,'gen3randombattle',True,False))
        cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,
            expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
        projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,
            expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
        cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog)
        cls.compiled=sub.compile_substitute(engine_catalog=cls.producer,metadata=cls.metadata)
        cls.account=execution.compile_accounting(cls.adapter)
        cls.actual=staticmethod(jax.jit(lambda state,actions,chance,tape:
            transition.step_with_chance(state,actions,cls.table,chance,_public_trace=tape,_lifecycle_trace=True,
                _substitute_context=sub.RegisteredContext(cls.compiled.device,cls.account.device))))

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def prefix(self,b=2):
        s=empty_state(b,seed=71)
        s=s._replace(species=j.ones_like(s.species).at[:,1].set(2),level=j.full_like(s.level,80),
            gender=j.ones_like(s.gender),hp=j.full_like(s.hp,101),max_hp=j.full_like(s.hp,101),
            stats=j.full_like(s.stats,200).at[...,0].set(101).at[:,0,:,5].set(300),
            active=j.broadcast_to(j.array([3,5],j.int32),(b,2)),moves=j.ones_like(s.moves),
            pp=j.full_like(s.pp,10),types=s.types.at[...,0].set(1),entry_pending=j.zeros_like(s.entry_pending),
            turn=j.zeros_like(s.turn),done=j.zeros_like(s.done))
        cursor=life.register(j.arange(201,201+b,dtype=j.int32),**self.kw)
        before=life.snapshot(cursor);t=life._tape(cursor,self.producer,64)
        for side in (0,1):t=life._reveal(t,s,j.full(b,side,j.int32),j.full(b,-1,j.int32),s.active[:,side],
            eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=True)
        t=t._replace(cursor=t.cursor._replace(initialized=j.ones(b,bool)));t=life._turn(t,j.ones(b,bool))
        frame=life.LifecycleResult(s,t.cursor,t,before,life.snapshot(t.cursor),j.zeros(b,bool),j.zeros(b,j.int32))
        owners=[]
        for owner in (0,1):
            stream=full.register(t.stream_id,j.full(b,owner,j.int32),compiled=self.adapter)
            current=v4.initialize(self.catalog,stream_ids=t.stream_id,owner_sides=j.full(b,owner,j.int32))
            consumed=full.project_and_fold(stream,frame,self.adapter.device,current,self.catalog)
            self.assertEqual(consumed.projection.error.tolist(),[0]*b);owners.append((consumed.projection.state,consumed.public))
        return s,t.cursor,owners

    def chance(self,b):
        return transition.ChanceInputs(j.zeros((b,2,12),j.int32).at[:,:,4].set(1),
            *[j.zeros((b,2),j.int32) for _ in range(4)],j.zeros(b,j.int32))

    def observed(self,s,cursor):
        t=life._tape(cursor,self.producer,64)
        ledger=execution.initialize(cursor.stream_id,compiled=self.account,capacity=16)
        return execution.begin_on_tape(t,ledger,self.account.device,active=j.ones(len(cursor.stream_id),bool))

    def test_actual_positive_literal_writes_two_genuine_owners_stage3_and_default_rng(self):
        s,cursor,owners=self.prefix();t=self.observed(s,cursor)
        acts=j.zeros((2,2),j.int32);chance=self.chance(2)
        out=self.actual(s,acts,chance,t)
        default=transition.step_with_chance(s,acts,self.table,chance)
        self.same(out.step,default)
        tape=out.callbacks;ledger=tape.execution.ledger
        self.assertEqual(ledger.count.tolist(),[2,2]);self.assertEqual(ledger.stage[:,:2].tolist(),[[2,2],[2,2]])
        for row in range(2):
            raw=np.asarray(tape.words[row,:int(tape.count[row])]);writes=raw[raw[:,0]==sub.SUBSTITUTE_CREATED]
            self.assertEqual(writes[:,1:3].tolist(),[[0,25],[0,25]])
            self.assertEqual(writes[:,5:9].tolist(),[[101,76,101,0],[101,76,101,0]])
        frame=life.LifecycleResult(out.step.state,tape.cursor,tape,life.snapshot(cursor),life.snapshot(tape.cursor),
            out.step.invalid_actions,out.step.placeholder_moves)
        drained=[]
        for owner,(stream,current) in enumerate(owners):
            closed,consumed=execution.consume_frame_device(ledger,frame,stream,current,active=j.ones(2,bool),
                tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog)
            self.assertEqual(consumed.projection.error.tolist(),[0,0]);drained.append(closed)
            # Original literal writer, both visibility paths; no emitter patch.
            for row in range(2):
                own='76/101' if owner==0 else '76/100'
                peer='76/101' if owner==1 else '76/100'
                prefix=['|switch|p1a: Snorlax|Snorlax, L80, M|101/101' if owner==0 else '|switch|p1a: Snorlax|Snorlax, L80, M|100/100',
                    '|switch|p2a: Absol|Absol, L80, M|101/101' if owner==1 else '|switch|p2a: Absol|Absol, L80, M|100/100','|turn|1']
                wire,_=public_wire.project_lines(public_wire.WireState(),prefix,**v4.wire_kwargs(self.catalog))
                lines=['|move|p1a: Snorlax|Substitute|p1a: Snorlax','|-start|p1a: Snorlax|Substitute',f'|-damage|p1a: Snorlax|{own}',
                    '|move|p2a: Absol|Substitute|p2a: Absol','|-start|p2a: Absol|Substitute',f'|-damage|p2a: Absol|{peer}','|','|upkeep','|turn|2']
                _,ps=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog))
                expected=np.array([e.words() for p in ps for e in p.events],np.int32)
                actual=np.asarray(consumed.projection.frames.words).transpose(1,0,2,3)[row][np.asarray(consumed.projection.frames.valid).transpose(1,0,2)[row]]
                np.testing.assert_array_equal(actual,expected)
            self.assertFalse(bool(consumed.projection.B4_qualified.any()))
        joined,good=execution.join_consumed_owners(*drained,tables=self.account.device)
        self.assertEqual(good.tolist(),[True,True]);self.assertEqual(joined.stage[:,:2].tolist(),[[3,3],[3,3]])
