"""Small current-cutpoint controls; no engine session/model/game execution."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_no_damage_callbacks as nd,engine_kernel_frame_risk as risk,engine_execution_completeness as ex
from gpu_engine import absorption,variable_moves,public_wire,v4_encoder
import test_gpu_engine_no_damage_owning as fixture
import gpu_engine_shared_public_jits as shared

class NoDamageGuards(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.NoDamageOwningTests.setUpClass();cls.f=fixture.NoDamageOwningTests()
  # A separate ORIGINAL host catalog adds canonical Hidden Power source rows;
  # it is not a copied binding or an injected prepared Context.
  from pokezero.dex import ShowdownDex,load_showdown_dex_cached
  from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
  from gpu_engine import public_belief,engine_public_events as events,engine_request_source as request,engine_own_facts as own
  from gpu_engine import engine_direct_hit_controls as hit,engine_leftovers_callbacks as leftovers,engine_retained_encoder_driver as driver
  from _showdown_root import showdown_root
  f=cls.f;original=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves)
  moves.update({name:original.move_info(name)for name in ('hiddenpower','hiddenpowerice')})
  dex=ShowdownDex(moves,f.dex.species,f.dex.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,tuple(Gen3RandbatVariant(name+str(i),'authored',name,'authored',80,
      ('Hidden Power Ice','Hydro Pump','Flamethrower','Thunderbolt'),abi,'Leftovers')for i,abi in enumerate(('Pressure','Water Absorb','Volt Absorb','Flash Fire'))))for name in f.metadata['vocabulary']['species'][1:]}
  src=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-hidden-power-no-damage'),variants)
  cat=v4_encoder.compile_encoder(public_belief.compile_source(src),dex,f.catalog.vocabulary,checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  producer=events.compile_catalog(f.metadata,f.table,public_moves=cat.moves,dex=dex,exposure=events.HealthExposure(events.SHOWDOWN,'gen3randombattle',True,False))
  rules=request._checked(f.req)['rules'];req=request.compile_request_source(f.metadata,vocabulary=cat.vocabulary,dex=dex,checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=producer)
  facts=own.compile_own_facts(req,cat);helper=hit.compile_direct_hit_controls(engine_catalog=producer)
  prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',
      leftovers_compiled=leftovers.compile_leftovers(f.metadata,engine_catalog=producer),leftovers_mode='retained',direct_hit_compiled=helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.dynamic=fixture.NoDamageOwningTests();cls.dynamic.catalog=cat;cls.dynamic.producer=producer;cls.dynamic.helper=helper;cls.dynamic.prepared=prepared;cls.dynamic.runtime=driver._checked(prepared)
  cls.dynamic.kw=dict(catalog=producer,expected_binding=producer.binding,expected_engine_binding=np.asarray(producer.engine_binding,np.uint32))
 def test_dynamic_member_type_and_actual_tryhit_provenance_no_public_type_label(self):
  f=self.dynamic;hp_id=f.metadata['vocabulary']['moves'].index('hiddenpowerice');data=f.helper.device.no_damage
  self.assertTrue(bool(data.candidates[hp_id]));self.assertTrue(bool(data.dynamic_type[hp_id]))
  for reason in (nd.MISS,nd.TYPE_IMMUNE,nd.WATER_HEAL,nd.VOLT_FULL,nd.FLASH_START):
   with self.subTest(reason=reason):
    tape,p,w,o,actor_hp,vol,sub,_,_=f.component(reason=reason);move=j.full(8,hp_id,j.int32)
    member_type=p.before.move_type;member_power=j.full(8,70,j.int32)
    power,typ,_=variable_moves.resolve(f.table.variable_moves,move,f.table.power[move],f.table.type[move],f.table.physical[move],
        member_type,member_power,j.full(8,255,j.int32),actor_hp,actor_hp)
    p=p._replace(move=move,before=p.before._replace(move_type=typ,actor_hp_type=member_type,actor_hp_power=member_power))
    original=absorption.try_hit(f.table.absorption,p.before.ability,move,typ,p.before.status,j.ones((8,2),j.int32),
        p.before.hp,p.before.maximum,p.before.sub_hp,p.before.volatile,eligible=p.requested&p.before.accuracy_hit)
    p=nd.observed_absorption(p,original);w=w._replace(power=power);o=o._replace(power=power)
    tape=tape._replace(words=tape.words.at[:,0,5].set(move))
    result=nd.finish(tape,p,w,o,actor_hp,vol,sub,member_type,member_power)
    captured=nd.Captured(p,w,actor_hp,vol,sub,member_type,member_power)
    self.assertTrue(bool(nd.captured_matches(result,captured,o).all()))
    self.assertFalse(bool(nd.captured_matches(result,captured._replace(actor_hp_type=member_type+1),o).any()))
    self.assertFalse(bool(nd.captured_matches(result,captured._replace(actor_hp_power=member_power-1),o).any()))
    # The original canonical map emits only Hidden Power, regardless of the
    # engine alias and actual member-derived attack type.
    public_id=f.producer.ids[hp_id];self.assertEqual(f.catalog.moves.names[public_id],'hiddenpower')
    self.assertTrue(bool(f.runtime['public'].device.no_damage_miss[hp_id]))
 def test_current_clocks_prefix_old_bits_and_malformed_global8(self):
  tape,p,w,o,hp,vol,sub,hpt,hpp=self.f.component(reason=nd.MISS)
  good=nd.finish(tape,p,w,o,hp,vol,sub,hpt,hpp);capture=nd.Captured(p,w,hp,vol,sub,hpt,hpp)
  for name,bad in (
    ('prefix',good._replace(words=good.words.at[0,0,5].add(1))),
    ('turn',good._replace(cursor=good.cursor._replace(turn=good.cursor.turn.at[0].add(1)))),
    ('generation',good._replace(event_generation=good.event_generation.at[0,2,1].add(1))),
    ('physical',good._replace(cursor=good.cursor._replace(physical_active=good.cursor.physical_active.at[0,1].set(5)))),
    ('claim',good._replace(execution=good.execution._replace(ledger=good.execution.ledger._replace(claimed=good.execution.ledger.claimed.at[0,2].set(0)))))):
   with self.subTest(name=name):self.assertEqual(nd.captured_matches(bad,capture,o).tolist(),[False]+[True]*7)
  for key,value in (('hit',j.ones(8,bool)),('immune',j.ones(8,bool)),('absorbed',j.ones(8,bool)),('critical',j.ones(8,bool)),('recoil',j.ones(8,j.int32)),('explodes',j.ones(8,bool))):
   with self.subTest(key=key):self.assertFalse(bool(nd.captured_matches(good,capture,o._replace(**{key:value})).any()))
  carry=good.execution.risks.carry._replace(risk=j.full(8,risk.BOOST,j.int32))
  prior=good._replace(refusal=j.full(8,65,j.int32),execution=good.execution._replace(risks=good.execution.risks._replace(carry=carry)))
  result=risk.outcome_on_tape(prior,o,eligible=j.ones(8,bool),active=j.ones(8,bool),captured_no_damage=capture)
  self.assertEqual(result.execution.risks.carry.risk.tolist(),[risk.BOOST|risk.UPSTREAM]*8);self.assertTrue(bool(((result.refusal&65)==65).all()))
  for field in ('binding','engine_binding','source_binding','move_binding','producer_binding'):
   original=getattr(p.context.data.no_damage,field)
   for bad in (original[None,:],j.tile(original,(8,1)),original.astype(j.int32)):
    changed=p._replace(context=p.context._replace(data=p.context.data._replace(no_damage=p.context.data.no_damage._replace(**{field:bad}))))
    with self.subTest(field=field,shape=bad.shape),self.assertRaises(ValueError):nd.finish(tape,changed,w,o,hp,vol,sub,hpt,hpp)
  foreign=p._replace(context=p.context._replace(data=p.context.data._replace(no_damage=p.context.data.no_damage._replace(binding=p.context.data.no_damage.binding.at[0].add(j.uint32(1))))))
  self.assertFalse(bool(nd.captured_matches(good,capture._replace(pending=foreign),o).any()))
 def test_unsupported_outcomes_inactive_and_full_capacity_never_get_empty_credit(self):
  mask=j.array([True]*4+[False]*4);t,p,w,o,hp,vol,sub,hpt,hpp=self.f.component(reason=nd.MISS,mask=mask)
  result=jax.jit(nd.finish)(t,p,w,o,hp,vol,sub,hpt,hpp)
  self.assertEqual(result.count.tolist(),[3]*4+[0]*4)
  for before,after in zip(jax.tree.leaves(t),jax.tree.leaves(result),strict=True):
   if before.ndim and before.shape[0]==8 and before.shape!=(8,):np.testing.assert_array_equal(before[4:],after[4:])
  for key,val in (('power',j.zeros(8,j.int32)),('protected',j.ones(8,bool)),('enduring',j.ones(8,bool)),('substitute',j.ones(8,bool))):
   bad=nd.finish(t,p,w._replace(**{key:val}),o,hp,vol,sub,hpt,hpp)
   self.assertFalse(bool((bad.execution.ledger.stage[:,0]==2).any()))
  for flag in (True,False):
   altered=p._replace(before=p.before._replace(accuracy_hit=j.full(8,flag),explodes=j.ones(8,bool)))
   self.assertFalse(bool(nd.scheduled(altered,w).any()))
  # A full tape with a valid immediate original MOVE prefix still reports
  # capacity overflow; clipped padding must never inspect/overwrite that prefix.
  t,p,w,o,hp,vol,sub,hpt,hpp=self.f.component(reason=nd.MISS)
  ledger=t.execution.ledger._replace(start=t.execution.ledger.start.at[:,0].set(64),end=t.execution.ledger.end.at[:,0].set(64))
  full=t._replace(count=j.full(8,64,j.int32),words=t.words.at[:,63].set(t.words[:,0]),
      event_turn=t.event_turn.at[:,63].set(t.event_turn[:,0]),event_phase=t.event_phase.at[:,63].set(t.event_phase[:,0]),event_generation=t.event_generation.at[:,63].set(t.event_generation[:,0]),
      switch_details=t.switch_details.at[:,63,0].set(9),execution=t.execution._replace(ledger=ledger))
  got=nd.finish(full,p,w,o,hp,vol,sub,hpt,hpp)
  self.assertTrue(bool(((got.refusal&nd.public.OVERFLOW)!=0).all()));np.testing.assert_array_equal(got.words,full.words)
  self.assertFalse(bool((got.execution.ledger.stage[:,0]==2).any()))
 def test_original_full_templates_keep_named_ability_and_lexical_scope(self):
  f=self.f;cat=f.catalog;name=cat.species.names[1];ident=('p1a: '+name,'p2a: '+name);kw=v4_encoder.wire_kwargs(cat)
  prior,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ident[0]}|{name}, L80, M|50/100',f'|switch|{ident[1]}|{name}, L80, M|100/100','|turn|1'],**kw)
  for reason,line in ((nd.MISS,f'|-miss|{ident[0]}|{ident[1]}'),(nd.TYPE_IMMUNE,f'|-immune|{ident[0]}'),
    (nd.WATER_HEAL,f'|-heal|{ident[0]}|71/100|[from] ability: Water Absorb|[of] {ident[1]}'),
    (nd.VOLT_HEAL,f'|-heal|{ident[0]}|71/100|[from] ability: Volt Absorb|[of] {ident[1]}'),
    (nd.WATER_FULL,f'|-immune|{ident[0]}|[from] ability: Water Absorb'),(nd.VOLT_FULL,f'|-immune|{ident[0]}|[from] ability: Volt Absorb'),
    (nd.FLASH_START,f'|-start|{ident[0]}|ability: Flash Fire'),(nd.FLASH_REPEAT,f'|-immune|{ident[0]}|[from] ability: Flash Fire')):
   _,projection=public_wire.project_line(prior,line,**kw);expected=np.array([e.words()for e in projection.events],np.int32)
   key=int(f.runtime['public'].device.no_damage[reason,0]);actual=np.asarray(f.runtime['public'].device.templates[key])[:len(expected)]
   np.testing.assert_array_equal(actual,expected)

 def test_all_literal_forms_both_original_full_folds_and_exact_writer_words(self):
  from gpu_engine import engine_full_v4_event_device as full,engine_lifecycle_callbacks as life,public_inputs as inputs
  f=self.f;b=2;kw=v4_encoder.wire_kwargs(f.catalog)
  fold=jax.jit(lambda stream,frame,public:full.project_and_fold(stream,frame,f.runtime['public'].device,public,f.catalog))
  fold_chunk=shared.fold(f.catalog)   # one compile; eagerly each call compiled its scan again (16 x ~46 s)
  namespaces={k:np.asarray(getattr(f.runtime['public'].device.namespaces,k))for k in full._NAMES}
  for reason in range(1,9):
   t,p,w,o,hp,vol,sub,hpt,hpp=f.component(b,reason=reason);out=nd.finish(t,p,w,o,hp,vol,sub,hpt,hpp)
   cursor=t.cursor._replace(revealed_species_id=j.broadcast_to(j.array([1,2],j.int32),(b,2)));out=out._replace(cursor=cursor)
   names=[f.catalog.species.names[int(f.runtime['public'].device.causal.species[i])]for i in (1,2)]
   ident=[f'p{i+1}a: '+names[i]for i in (0,1)]
   prefix=[f'|switch|{ident[0]}|{names[0]}, L80, M|100/100',f'|switch|{ident[1]}|{names[1]}, L80, M|{int(p.before.hp[0])}/100','|turn|1']
   wire,packets=public_wire.project_lines(public_wire.WireState(),prefix,**kw)
   initial=[e for packet in packets for e in packet.events]
   move=f.catalog.dex.move_info(f.metadata['vocabulary']['moves'][int(p.move[0])]).name
   if reason==nd.MISS:line=f'|-miss|{ident[0]}|{ident[1]}'
   elif reason==nd.TYPE_IMMUNE:line='|-immune|'+ident[1]
   elif reason==nd.FLASH_START:line=f'|-start|{ident[1]}|ability: Flash Fire'
   elif reason in (nd.WATER_HEAL,nd.VOLT_HEAL):
    ability='Water Absorb'if reason==nd.WATER_HEAL else 'Volt Absorb'
    line=f'|-heal|{ident[1]}|75/100|[from] ability: {ability}|[of] {ident[0]}'
   else:line=f'|-immune|{ident[1]}|[from] ability: '+{nd.WATER_FULL:'Water Absorb',nd.VOLT_FULL:'Volt Absorb',nd.FLASH_REPEAT:'Flash Fire'}[reason]
   _,packets=public_wire.project_lines(wire,[f'|move|{ident[0]}|{move}|{ident[1]}',line],**kw)
   expected=np.array([e.words()for packet in packets for e in packet.events],np.int32)
   for owner in (0,1):
    with self.subTest(reason=reason,owner=owner):
     public=v4_encoder.initialize(f.catalog,stream_ids=t.stream_id,owner_sides=j.full(b,owner,j.int32))
     for chunk in inputs.pack_chunks(np.asarray(t.stream_id),[initial]*b,**namespaces):
      public,error=fold_chunk(public,chunk);self.assertFalse(bool(error.any()))
     stream=full.register(t.stream_id,j.full(b,owner,j.int32),compiled=f.runtime['public'])
     stream=stream._replace(causal=stream.causal._replace(cursor=cursor,sequence=j.full(b,3,j.int32)),sequence=public.cursor.sequence,
       live=j.ones((b,2),bool),species=public.mons.species,status=public.mons.status,hp=public.mons.hp_numerator,maximum=public.mons.hp_denominator)
     frame=life.LifecycleResult(None,cursor,out,life.snapshot(cursor),life.snapshot(cursor),j.zeros(b,bool),j.zeros(b,j.int32))
     got=fold(stream,frame,public);self.assertEqual(got.projection.error.tolist(),[0]*b)
     for row in (0,1):
      actual=np.asarray(got.projection.frames.words).transpose(1,0,2,3)[row][np.asarray(got.projection.frames.valid).transpose(1,0,2)[row]]
      np.testing.assert_array_equal(actual,expected)
