"""Causal entry weather -> original full public controls; tiny source fixtures."""
from copy import deepcopy
from dataclasses import replace
from pokezero.dex import ShowdownDex
import unittest
from types import MappingProxyType
import jax
import jax.numpy as j
import numpy as np
from pokezero.randbat import Gen3RandbatSource,RandbatSourceMetadata,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant
from pokezero.category_vocab import CategoryVocabulary
from gpu_engine import transition as t,engine_lifecycle_callbacks as life,engine_public_events as events
from gpu_engine import engine_public_event_device as base,engine_full_v4_event_device as full
from gpu_engine import engine_weather_callbacks as weather_callbacks,weather,public_wire,v4_encoder as v4
import test_gpu_engine_engine_status_callbacks as fixture


class WeatherPublicTests(unittest.TestCase):
    kwargs=fixture.StatusCallbacksTests.kwargs
    initial=fixture.StatusCallbacksTests.initial
    chance=fixture.StatusCallbacksTests.chance
    step=fixture.StatusCallbacksTests.step
    same=fixture.StatusCallbacksTests.same
    start=fixture.StatusCallbacksTests.start
    compare=fixture.StatusCallbacksTests.compare

    @classmethod
    def setUpClass(cls):
        fixture.StatusCallbacksTests.setUpClass();f=fixture.StatusCallbacksTests
        cls.f,cls.ids=f.f,dict(f.ids)
        cls.metadata=deepcopy(f.metadata)
        for name in ('drizzle','drought','sandstream','pressure','trace'):
            if name not in cls.metadata['vocabulary']['abilities']:cls.metadata['vocabulary']['abilities'].append(name)
        moves=dict(f.catalog.dex.moves)
        for name,typ in (('raindance','Water'),('sunnyday','Fire'),('sandstorm','Rock'),('hail','Ice')):
            cls.ids[name]=len(cls.metadata['moves'])
            cls.metadata['vocabulary']['moves'].append(name)
            cls.metadata['moves'].append(dict(id=name,base_power=0,
                type=cls.metadata['vocabulary']['types'].index(typ.lower()),source_type=typ,
                category='Status',pp=20,accuracy=True,priority=0,flags={}))
            moves[name]=replace(f.catalog.dex.moves['surf'],id=name,name=name,type=typ,category='Status',
                gen3_category='Status',base_power=0,pp=20,target='all',status=None)
        dex=ShowdownDex(moves,f.catalog.dex.species,f.catalog.dex.type_chart)
        names=cls.metadata['vocabulary']['species'][1:]
        variants={name:Gen3RandbatSpeciesUniverse(name,80,tuple(Gen3RandbatVariant(
            name+'-'+a,'fixture',name,'fixture',80,('Surf','Hidden Power Ice'),a,'Leftovers')
            for a in ('Drizzle','Drought','Sand Stream','Pressure','Trace'))) for name in names}
        host=public_belief_source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'weather-public-fixture'),variants)
        from gpu_engine import public_belief
        host=public_belief.compile_source(public_belief_source)
        labels=set(f.catalog.vocabulary.tokens)
        for name in ('drizzle','drought','sandstream','pressure','trace'):
            labels|={'ability:'+name,'belief:possible_ability:'+name}
        for name in cls.ids:labels|={'move:'+name,'belief:possible_move:'+name}
        labels|={'weather:raindance','weather:sunnyday','weather:sandstorm','weather:hail'}
        cls.catalog=v4.compile_encoder(host,dex,CategoryVocabulary(tuple(sorted(labels))),
            checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
        cls.table=t.tables_from_metadata(cls.metadata)
        cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.catalog.dex,
            exposure=events.HealthExposure(events.SHOWDOWN,'gen3randombattle',True,False))
        projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,
            expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
        cls.compiled=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog)
        cls.fast=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.compiled.device,p,cls.catalog)))

    def arrays(self,first='drizzle',second='drought',physical=0):
        arrays={k:np.array(v,copy=True) for k,v in self.f.arrays.items()}
        abilities=self.metadata['vocabulary']['abilities']
        arrays['ability'][:]=abilities.index('pressure')
        arrays['ability'][0,physical]=abilities.index(first)
        arrays['ability'][1,physical]=abilities.index(second)
        return arrays

    def test_actual_initial_weather_order_exact_both_owner_full_words_and_state(self):
        frame=self.initial(arrays=self.arrays())
        for row in range(2):
            words=np.asarray(frame.callbacks.words[row]);at=np.flatnonzero(words[:,0]==12)
            self.assertEqual(len(at),2)
            self.assertEqual(words[at[1],5],words[at[0],6])
        for owner in (0,1):
            s,p=self.start(frame,j.full(2,owner,j.int32))
            result,_=self.compare(s,frame,p,[public_wire.WireState()]*2)
            self.assertTrue(bool(result.public.context.field.weather_ability.all()))
            self.assertFalse(bool(result.projection.full_line_consumers_complete.any()))

    def test_actual_nonphysical_switch_weather_and_malformed_row_atomicity(self):
        frame=self.initial(arrays=self.arrays(physical=5))
        s,p=self.start(frame,j.array([0,1],j.int32))
        initial,wire=self.compare(s,frame,p,[public_wire.WireState()]*2)
        changed=self.step(frame,[[8,8],[8,8]])
        result,_=self.compare(initial.projection.state,changed,initial.public,wire)
        words=np.asarray(changed.callbacks.words[0]);at=np.flatnonzero(words[:,0]==12)
        self.assertEqual(len(at),2);self.assertEqual(words[at,2].tolist(),[5,5])
        for column,value in ((1,2),(2,0),(5,weather.HAIL),(7,0),(8,1),(9,0)):
            bad=changed._replace(callbacks=changed.callbacks._replace(words=changed.callbacks.words.at[0,int(at[0]),column].set(value)))
            failed=self.fast(initial.projection.state,bad,initial.public)
            self.assertNotEqual(int(failed.projection.error[0]),0)
            self.assertEqual(int(failed.projection.error[1]),0)
            self.assertFalse(bool(failed.projection.frames.valid[:,0].any()))
            self.assertEqual(int(failed.public.cursor.sequence[0]),int(initial.public.cursor.sequence[0]))
            self.same(full._commit_public(j.array([False,True]),initial.public,failed.public),failed.public)

    def test_actual_called_move_then_upkeep_and_expiry_exact_both_owners(self):
        frame=self.initial('raindance',arrays=self.arrays('pressure','pressure'))
        s,p=self.start(frame,j.array([0,1],j.int32))
        result,wire=self.compare(s,frame,p,[public_wire.WireState()]*2)
        for turn in range(5):
            # Turn1 actually calls Rain Dance; later Surf uses avoid the
            # deliberately still-unknown same-weather failed-move outcome.
            actions=[0,8] if turn==0 else ([8,4] if turn%2 else [4,8])
            changed=self.step(frame,[actions]*2)
            raw=np.asarray(changed.callbacks.words[0,:int(changed.callbacks.count[0])])
            self.assertEqual((raw[:,0]==13).sum(),int(turn==0))
            clock=raw[raw[:,0]==14]
            self.assertEqual(len(clock),1,msg=str(dict(turn=turn,actions=actions,
                weather=changed.state.weather.tolist(),counter=changed.state.weather_counter.tolist(),
                pending=changed.state.pending_residual.tolist(),raw=raw.tolist())))
            self.assertEqual(int(clock[0,9]),2 if turn==4 else 1)
            result,wire=self.compare(result.projection.state,changed,result.public,wire)
            self.assertEqual(result.projection.state.weather_remaining.tolist(),[max(0,4-turn)]*2)
            frame=changed
        self.assertEqual(result.public.context.field.weather.tolist(),[0,0])
        self.assertFalse(bool(result.public.context.field.weather_set.any()))

    def test_permanent_clock_requires_one_actual_residual_boundary_and_whole_row_rollback(self):
        frame=self.initial(arrays=self.arrays('drizzle','pressure'))
        s,p=self.start(frame,j.array([0,1],j.int32))
        initial=self.fast(s,frame,p)
        changed=self.step(frame,[[8,8],[8,8]])
        baseline=self.fast(initial.projection.state,changed,initial.public)
        self.assertEqual(baseline.projection.error.tolist(),[0,0])
        kinds=np.asarray(changed.callbacks.words[0,:,0])
        clock=int(np.flatnonzero(kinds==14)[0]);blank=int(np.flatnonzero(kinds==7)[0])
        self.assertGreater(clock,blank)
        for duplicate in (True,False):
            indices=np.arange(64)
            if duplicate:
                indices[clock+1:]=np.arange(clock,63)
            else:indices[clock],indices[blank]=indices[blank],indices[clock]
            updates={}
            for name in ('words','event_turn','event_phase','event_generation','switch_details'):
                a=np.array(getattr(changed.callbacks,name));a[0]=a[0,indices];updates[name]=j.array(a)
            updates['count']=changed.callbacks.count.at[0].add(int(duplicate))
            bad=changed._replace(callbacks=changed.callbacks._replace(**updates))
            failed=self.fast(initial.projection.state,bad,initial.public)
            self.assertNotEqual(int(failed.projection.error[0]),0)
            self.assertEqual(int(failed.projection.error[1]),0)
            self.assertFalse(bool(failed.projection.frames.valid[:,0].any()))
            self.assertEqual(int(failed.public.context.field.weather_upkeeps[0]),0)
            self.assertEqual(int(failed.public.cursor.sequence[0]),int(initial.public.cursor.sequence[0]))
            self.same(full._commit_public(j.array([False,True]),initial.public,failed.public),failed.public)

    def test_namespace_edge_drift_and_unattributed_copy_stay_closed(self):
        catalog=self.compiled.causal.weather_catalog
        before=catalog.field.weather_ids
        try:
            object.__setattr__(catalog.field,'weather_ids',MappingProxyType(dict(before,raindance=0)))
            with self.assertRaises(ValueError):full.check_adapter(self.compiled)
        finally:object.__setattr__(catalog.field,'weather_ids',before)
        arrays=self.arrays('trace','drizzle')
        frame=self.initial(arrays=arrays)
        self.assertTrue(bool((frame.callbacks.refusal & weather_callbacks.UNKNOWN_WEATHER).any()))
        s,p=self.start(frame,j.array([0,1],j.int32));result=self.fast(s,frame,p)
        self.assertFalse(bool(result.projection.frames.valid.any()))

    def test_actual_same_weather_failure_and_chip_stay_internal_unknown(self):
        frame=self.initial('raindance',arrays=self.arrays('pressure','pressure'))
        s,p=self.start(frame,j.array([0,1],j.int32));initial=self.fast(s,frame,p)
        first=self.step(frame,[[0,8],[0,8]])
        accepted=self.fast(initial.projection.state,first,initial.public)
        self.assertEqual(accepted.projection.error.tolist(),[0,0])
        repeated=self.step(first,[[0,4],[0,4]])
        raw=np.asarray(repeated.callbacks.words[0,:int(repeated.callbacks.count[0])])
        failure=raw[raw[:,0]==15]
        self.assertEqual(failure.tolist(),[[15,0,0,-1,-1,weather.RAIN,weather.RAIN,
            self.ids['raindance'],weather.RAIN,1]])
        self.assertTrue(bool((repeated.callbacks.refusal & weather_callbacks.UNKNOWN_WEATHER).all()))
        refused=self.fast(accepted.projection.state,repeated,accepted.public)
        self.assertFalse(bool(refused.projection.frames.valid.any()))
        self.same(refused.public,accepted.public)
        hail=self.initial('hail',arrays=self.arrays('pressure','pressure'))
        hs,hp=self.start(hail,j.array([0,1],j.int32));before=self.fast(hs,hail,hp)
        chipped=self.step(hail,[[0,8],[0,8]])
        raw=np.asarray(chipped.callbacks.words[0,:int(chipped.callbacks.count[0])])
        damage=raw[raw[:,0]==16];self.assertGreater(len(damage),0)
        for w in damage:
            self.assertGreater(w[5],w[6]);self.assertGreaterEqual(w[7],w[5])
            self.assertEqual(int(w[9]),weather.HAIL)
        self.assertTrue(bool((chipped.callbacks.refusal & weather_callbacks.UNKNOWN_WEATHER).all()))
        refused=self.fast(before.projection.state,chipped,before.public)
        self.assertFalse(bool(refused.projection.frames.valid.any()))
        self.same(refused.public,before.public)


if __name__=='__main__':unittest.main()
