"""Pinned pure primary-SLP input and diagnostic components, no scientific roots."""
from copy import deepcopy
from dataclasses import replace
import ast
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_sleep_powder_callbacks as sleep,engine_toxic_callbacks as primary
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_public_events as public,engine_leftovers_callbacks as leftovers
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import public_belief,source_provenance,transition,v4_encoder,mechanics as m
from gpu_engine import public_wire,engine_full_v4_event_device as full
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
import test_gpu_engine_sleep_powder_runtime as legacy
from _showdown_root import showdown_root

class PrimarySleepRuntimeTests(unittest.TestCase):
    same=legacy.SleepPowderRuntimeTests.same
    @classmethod
    def setUpClass(cls):
        legacy.SleepPowderRuntimeTests.setUpClass();f=legacy.SleepPowderRuntimeTests;cls.old=f
        cls.metadata=deepcopy(f.metadata);cls.original=load_showdown_dex_cached(showdown_root())
        moves=dict(f.catalog.dex.moves)
        for name in ('sleeppowder','hypnosis','spore','sing'):
            info=cls.original.move_info(name)
            if name not in cls.metadata['vocabulary']['moves']:
                cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
            i=cls.metadata['vocabulary']['moves'].index(name)
            cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),
                source_type=info.type,category=info.category,pp=info.pp,accuracy=int(info.accuracy),priority=info.priority,flags={'protect':1,'reflectable':1})
            moves[name]=info
        # Real source Dark typing is installed in the literal constructor, not
        # mutated into an existing Session or reconstructed from observations.
        cls.metadata['vocabulary']['species'].append('absol');species=dict(f.catalog.dex.species,absol=cls.original.species_info('absol'))
        cls.metadata['type_chart'][13][16]=0  # original Psychic -> Dark damage immunity, unused by SLP setter
        cls.dex=ShowdownDex(moves,species,cls.original.type_chart)
        cls.names=('spore','hypnosis','raindance','hail')
        variants={n:Gen3RandbatSpeciesUniverse(n,80,tuple(Gen3RandbatVariant('sleep-'+n+str(i),'authored',n,'authored',80,
            ms,'Pressure','Leftovers') for i,ms in enumerate((cls.names,('sleeppowder','sing','raindance','hail')))))
            for n in cls.metadata['vocabulary']['species'][1:]}
        src=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-primary-sleep-fixture'),variants)
        labels=set(f.catalog.vocabulary.tokens)|{'species:absol','type:Dark'}
        for name in ('sleeppowder','hypnosis','spore','sing'):labels|={'move:'+name,'belief:possible_move:'+name}
        cls.catalog=v4_encoder.compile_encoder(public_belief.compile_source(src),cls.dex,CategoryVocabulary(tuple(sorted(labels))),
            checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
        cls.table=transition.tables_from_metadata(cls.metadata)
        cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,
            exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
        cls.rules=request.load_source_rules(showdown_root(),cls.metadata)
        cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,
            checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer)
        cls.facts=own.compile_own_facts(cls.req,cls.catalog)
        cls.move_id=cls.metadata['vocabulary']['moves'].index('hypnosis')
        cls.helper=sleep.compile_primary_sleep(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
        cls.prepared=driver.prepare(cls.req,cls.facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,
            frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),
            leftovers_mode='retained',sleep_powder_compiled=cls.helper,sleep_powder_mode='kernel-event',
            fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
        cls.runtime=driver._checked(cls.prepared)

    def test_original_descriptor_mask_explicit_binding_and_default_program(self):
        self.assertEqual(self.runtime['public'].device.primary_sleep_status.shape,(len(self.table.power),))
        self.assertEqual(self.runtime['public'].device.causal.primary_sleep_moves.shape,(len(self.table.power),))
        ids=self.metadata['vocabulary']['moves'];chosen=[ids[i] for i in np.flatnonzero(self.helper.device.move_mask)]
        self.assertEqual(set(chosen),{'sleeppowder','hypnosis','spore','sing'})
        self.assertTrue(bool(self.helper.device.primary_sleep));self.assertEqual(int(self.helper.device.move_id),0)
        self.assertEqual(int(self.old.helper.device.move_id),self.old.move_id)
        self.assertEqual(np.flatnonzero(self.old.helper.device.move_mask).tolist(),[self.old.move_id])
        self.assertFalse(bool(self.old.helper.device.primary_sleep))
        for name in ('rest','sleeptalk','toxic','surf','transform'):
            if name in ids:self.assertFalse(bool(self.helper.device.move_mask[ids.index(name)]))
        for value in (sleep.Compiled(self.helper.device),replace(self.helper)):
            with self.assertRaises(ValueError):sleep.check_compiled(value)
        with self.assertRaises(ValueError):jax.jit(lambda:sleep.check_compiled(self.helper).binding)()
        fake=request.SourceRules(self.rules.device)
        with self.assertRaises(ValueError):sleep.compile_primary_sleep(engine_catalog=self.producer,metadata=self.metadata,source_rules=fake)
        before=ast.parse(subprocess.check_output(['git','show','7c6c6042:gpu_engine/transition.py'],text=True));after=ast.parse(Path('gpu_engine/transition.py').read_text())
        baseline=dump_default_program(before);self.assertEqual(dump_default_program(after),baseline);assert_hp_rng_mutations_detected(self,after,baseline)
        np.testing.assert_array_equal(self.runtime['kernel_risk_compiled'].device.sleep_powder_candidates,self.helper.device.move_mask)
        closed=risk.compile_risks(self.producer,self.runtime['execution_compiled'])
        for name in chosen:self.assertNotEqual(int(closed.device.selected_risk[ids.index(name)])&risk.NON_WEATHER,0)

    def test_exact_original_writer_templates_and_literal_unused_immunity(self):
        cat=self.catalog;name=cat.species.names[1];ident=('p1a: '+name,'p2a: '+name);kwargs=v4_encoder.wire_kwargs(cat)
        prior,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ident[0]}|{name}, L80, M|100/100',f'|switch|{ident[1]}|{name}, L80, M|100/100','|turn|1'],**kwargs)
        for move in np.flatnonzero(self.helper.device.move_mask):
            info=self.dex.move_info(self.metadata['vocabulary']['moves'][move]);_,p=public_wire.project_line(prior,f'|-status|{ident[0]}|slp|[from] move: {info.name}',**kwargs)
            key=int(self.runtime['public'].device.primary_sleep_status[move]);expected=np.array([e.words()for e in p.events],np.int32)
            self.assertGreater(key,0);np.testing.assert_array_equal(np.asarray(self.runtime['public'].device.templates[key])[:len(expected)],expected)
        b=8;z=j.zeros(b,j.int32);yes=j.ones(b,bool);no=j.zeros(b,bool);move=j.full(b,self.metadata['vocabulary']['moves'].index('hypnosis'),j.int32)
        attempt=sleep.Attempt(yes,yes,no,no,no,yes,yes,no,no,yes,no,z,j.full(b,m.SLP,j.int32),z,j.full(b,2,j.int32))
        raw=primary.words_device(z,z,z,move,attempt)
        self.assertEqual(sleep.supported_raw(raw,primary_sleep=True).tolist(),[True]*b)
        self.assertEqual(sleep.supported_raw(raw).tolist(),[False]*b)
        for field,value in (('applied',no),('accuracy_hit',no),('hit',no),('type_immune',yes),('ability_immune',yes),('substitute_block',yes),('protected',yes),('absorbed',yes),('secondary_active',yes),('old_status',j.full_like(z,m.SLP)),('old_counter',j.ones_like(z))):
            with self.subTest(field=field):self.assertFalse(bool(sleep.supported_raw(primary.words_device(z,z,z,move,attempt._replace(**{field:value})),primary_sleep=True).any()))
        for count in (0,1,6,15):
            self.assertFalse(bool(sleep.supported_raw(primary.words_device(z,z,z,move,attempt._replace(new_counter=j.full_like(z,count))),primary_sleep=True).any()))

    test_current_source_span_b8_and_lum_scope_stay_closed=legacy.SleepPowderRuntimeTests.test_current_source_span_b8_and_lum_scope_stay_closed
    test_all_tape_global_namespaces_require_exact_words_before_comparison=legacy.SleepPowderRuntimeTests.test_all_tape_global_namespaces_require_exact_words_before_comparison

    def test_actual_spore_and_hypnosis_dark_both_owner_frame_and_diagnostics(self):
        arrays=self.old.base().arrays('pressure','pressure');ids=self.metadata['vocabulary']['moves'];types=self.metadata['vocabulary']['types']
        arrays['moves'][:]=np.array([ids.index(n)for n in self.names],np.int32);arrays['pp'][:]=np.array([24,32,8,16],np.int32)
        arrays['species'][1,4]=self.metadata['vocabulary']['species'].index('absol');arrays['types'][1,4]=[types.index('dark'),0]
        with TemporaryDirectory()as tmp:
            p=Path(tmp)/'literal.npz';np.savez_compressed(p,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(p);session=driver.construct(pool,np.array([[0,1],[0,1]],np.int64),prepared=self.prepared,stream_ids=j.array([12901,12902],j.int32),seed=31)
        self.assertEqual(session.state.core.admitted.tolist(),[True,True]);before=session.state.core.engine
        draws=j.zeros((2,2,12),j.int32).at[:,:,4].set(1);chance=transition.ChanceInputs(draws,*[j.zeros((2,2),j.int32)for _ in range(4)],j.zeros(2,j.int32))
        actions=j.array([[0,7],[1,7]],j.int32);frames=[];original=self.runtime['frame']
        def capture(*args,**kwargs):
            result=original(*args,**kwargs);frames.append(result);return result
        self.runtime['frame']=capture
        try:result=driver.advance(session,actions,chance_inputs=chance)
        finally:self.runtime['frame']=original
        self.same(result.state.core.engine,transition.step_with_chance(before,actions,self.table,chance).state)
        frame=frames[0];report=result.state.core.frame_input_report
        diagnostic=dict(risk=result.state.core.risks.risk.tolist(),refusal=frame.callbacks.refusal.tolist(),ledger=result.state.core.execution.refusal.tolist(),report=report.refusal.tolist(),owners=[o.reason.tolist()for o in result.owners])
        self.assertEqual(result.state.core.admitted.tolist(),[True,True],json.dumps(diagnostic))
        raw=frame.callbacks.words;present=j.arange(raw.shape[1])[None,:]<frame.callbacks.count[:,None]
        for kind in (9,26):self.assertEqual(j.sum(present&(raw[:,:,0]==kind),axis=1).tolist(),[1,1])
        at=j.argmax(present&(raw[:,:,0]==26),axis=1);marker=raw[j.arange(2),at]
        self.assertEqual(marker[:,5].tolist(),[ids.index('spore'),ids.index('hypnosis')]);self.assertEqual(marker[:,7:10].tolist(),[[0,m.SLP,32]]*2)
        self.assertTrue(int(marker[1,6])&primary.DAMAGE_IMMUNE);self.assertFalse(int(marker[1,6])&primary.SECONDARY)
        ledger=result.state.core.execution;selected=(ledger.site==int(ex.Site.PRIMARY_SLEEP))&(ledger.serial>0)
        self.assertEqual(selected.sum(axis=1).tolist(),[1,1]);self.assertTrue(bool(j.all(j.where(selected,ledger.stage==3,True))))
        self.assertEqual(result.state.core.risks.risk.tolist(),[0,0])
        for owner in result.owners:
            self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True]);self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True]);self.assertFalse(bool(owner.B4_qualified.any()))
        from gpu_engine import engine_outcome_diagnostics as diag
        values=diag.refused_rows(report.outcome_diagnostics,np.array([False,True]))
        self.assertEqual([r['row']for r in values['rows']],[1]);row=values['rows'][0];self.assertTrue(row['available']);self.assertFalse(row['overflow'])
        actual=[v for v in row['records']if v['eligible']]
        self.assertEqual(len(actual),1);self.assertEqual(actual[0]['move'],ids.index('hypnosis'));self.assertEqual(actual[0]['immune'],1)
        self.assertEqual(actual[0]['actor_side'],0);self.assertEqual(actual[0]['target_physical'],4)
