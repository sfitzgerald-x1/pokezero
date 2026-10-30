"""Conditional nonlethal weather HP formatter; tiny source fixtures, no models."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_weather_speed_profile as profile,engine_request_source as source
from gpu_engine import engine_full_v4_event_device as full,public_wire,v4_encoder as v4,public_inputs as inputs
from gpu_engine import engine_public_events as events,weather
import test_gpu_engine_engine_weather_public as fixture


class WeatherHPPublicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.WeatherPublicTests.setUpClass();cls.f=fixture.WeatherPublicTests
        cls.rules=source.load_source_rules(cls.f.f.f.root,cls.f.metadata,node=cls.f.f.f.node)
        cls.profile=profile.compile_profile(cls.f.metadata,catalog=cls.f.catalog,engine_catalog=cls.f.producer,rules=cls.rules)
        cls.fast=staticmethod(jax.jit(lambda state,frame,public,record:full.project_and_fold(
            state,frame,cls.f.compiled.device,public,cls.f.catalog,weather_hp=(record,cls.profile.device))))

    def prepared(self):
        f=self.f();initial=f.initial('hail',arrays=f.arrays('pressure','pressure'))
        state,public=f.start(initial,j.array([0,1],j.int32))
        accepted,wire=f.compare(state,initial,public,[public_wire.WireState()]*2)
        # Trusted lower-primitive fixture, NOT an actual Session origin claim.
        # The actual source callsite stays UNKNOWN until separate integration.
        birth=initial.state._replace(turn=j.zeros(2,j.int32))
        record=profile.qualify_birth(birth,initial.cursor.stream_id,profile=self.profile)
        self.assertEqual(record.qualified_birth.tolist(),[True,True])
        frame=f.step(initial,[[0,8],[0,8]])
        words=np.array(frame.callbacks.words);meta={}
        for row in range(2):
            ids=np.flatnonzero(words[row,:,0]==16)
            self.assertEqual(len(ids),2)
            order=ids[np.argsort([-words[row,k,1+words[row,k,3]] for k in ids],kind='stable')]
            for name in ('words','event_turn','event_phase','event_generation','switch_details'):
                if name not in meta:meta[name]=np.array(getattr(frame.callbacks,name))
                meta[name][row,ids]=np.asarray(getattr(frame.callbacks,name))[row,order]
        tape=frame.callbacks._replace(**{k:j.array(v) for k,v in meta.items()},
            refusal=frame.callbacks.refusal & ~j.array(256,j.int32))
        return accepted,wire,frame._replace(callbacks=tape),record

    def test_both_owners_exact_all_words_and_public_leaves_with_source_tag(self):
        before,wire,frame,record=self.prepared()
        lower=full.causal.project_frame(before.projection.state.causal,frame,self.f.compiled.causal.device,
            weather_hp=(record,self.profile.device));self.assertEqual(lower.error.tolist(),[0,0])
        rows=[];next_wire=[]
        for row in range(2):
            active=dict(wire[row].active_idents());lines=[]
            sem=np.array(lower.chunk.words[row])[np.array(lower.chunk.valid[row])]
            for index,w in enumerate(sem):
                kind,_,turn,_,actor,_,target,_,*p=map(int,w);raw=np.asarray(frame.callbacks.words[row,index])
                if kind==inputs.Kind.SWITCH:
                    name=self.f.catalog.species.names[p[1]];active[target]=f'p{target+1}a: {name}'
                    lines.append(f'|switch|{active[target]}|{name}, L{p[3]}'+(', M' if p[4]==1 else ', F' if p[4]==2 else '')+f'|{p[5]}/{p[6]}')
                elif kind==inputs.Kind.MOVE:lines.append(f'|move|{active[actor]}|{self.f.catalog.moves.names[p[1]]}|{active[target]}')
                elif kind==inputs.Kind.FIELD:lines.append('|-weather|Hail'+('|[upkeep]' if raw[0]==14 else ''))
                elif kind==inputs.Kind.CONDITION:
                    token=('','slp','par','brn','frz','psn','tox')[p[3]]
                    lines.append(f'|-damage|{active[target]}|{p[1]}/{p[2]}'+(' '+token if token else '')+'|[from] Hail')
                elif kind==inputs.Kind.ACTION_CHUNK_END:lines.append('|')
                elif kind==inputs.Kind.UPKEEP:lines.append('|upkeep')
                elif kind==inputs.Kind.TURN:lines.append(f'|turn|{turn}')
                else:self.fail('unexpected kind '+str(kind))
            new,projections=public_wire.project_lines(wire[row],lines,**v4.wire_kwargs(self.f.catalog))
            rows.append([e for p in projections for e in p.events]);next_wire.append(new)
        namespaces={k:np.array(getattr(self.f.compiled.device.namespaces,k)) for k in full._NAMES if k!='source_binding'}
        expected=list(inputs.pack_chunks(np.array(frame.cursor.stream_id),rows,
            source_binding=np.array(self.f.catalog.source.device.binding),**namespaces))
        want=before.public
        for chunk in expected:
            want,error=v4.fold_chunk(want,chunk,self.f.catalog);self.assertEqual(error.tolist(),[0,0])
        got=self.fast(before.projection.state,frame,before.public,record)
        pre=full.project_frame(before.projection.state,frame,self.f.compiled.device,weather_hp=(record,self.profile.device))
        self.assertEqual(got.projection.error.tolist(),[0,0],msg=str(dict(pre=pre.error.tolist(),
            raw=np.asarray(frame.callbacks.words[0,:int(frame.callbacks.count[0])]).tolist(),
            hp=before.projection.state.hp.tolist(),maximum=before.projection.state.maximum.tolist())))
        self.f().same(got.public,want)
        for i,chunk in enumerate(expected):
            self.f().same(got.projection.frames.words[i],chunk.words)
            self.f().same(got.projection.frames.valid[i],chunk.valid)
        self.assertFalse(bool(got.projection.full_encoder_admitted.any()))
        self.assertFalse(bool(got.projection.B4_qualified.any()))

    def test_missing_birth_foreign_binding_old_hp_status_order_and_duplicate_refuse(self):
        before,_,frame,record=self.prepared();state=before.projection.state
        closed=full.project_frame(state,frame,self.f.compiled.device)
        self.assertTrue(bool((closed.error!=0).all()))
        ids=np.flatnonzero(np.array(frame.callbacks.words[0,:,0])==16);first,last=map(int,ids)
        for label in ('birth','binding','old_hp','status','order','duplicate','lethal','weather'):
            words=frame.callbacks.words;r=record
            if label=='birth':r=r._replace(qualified_birth=r.qualified_birth.at[0].set(False))
            elif label=='binding':r=r._replace(profile_binding=r.profile_binding.at[0,0].add(j.uint32(1)))
            elif label=='old_hp':words=words.at[0,first,5].add(-1)
            elif label=='status':words=words.at[0,first,8].set(1)
            elif label=='order':words=words.at[0,first].set(frame.callbacks.words[0,last]).at[0,last].set(frame.callbacks.words[0,first])
            elif label=='duplicate':words=words.at[0,last].set(words[0,first])
            elif label=='lethal':words=words.at[0,first,6].set(0)
            else:words=words.at[0,first,9].set(weather.SAND)
            bad=frame._replace(callbacks=frame.callbacks._replace(words=words))
            got=self.fast(state,bad,before.public,r)
            self.assertNotEqual(int(got.projection.error[0]),0,msg=label)
            self.assertEqual(int(got.projection.error[1]),0,msg=label)
            self.assertFalse(bool(got.projection.frames.valid[:,0].any()),msg=label)
            self.assertEqual(int(got.public.cursor.sequence[0]),int(before.public.cursor.sequence[0]),msg=label)


if __name__=='__main__':unittest.main()
