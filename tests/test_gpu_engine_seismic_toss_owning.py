"""Fixed-level source input at actual write; no worlds/models/Native/games."""
import ast,json,subprocess,shutil,unittest,os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import transition as tr,engine_direct_hit_controls as direct,engine_kernel_frame_risk as risks
from gpu_engine import engine_execution_completeness as ex,engine_public_events as events,engine_lifecycle_callbacks as life
from gpu_engine import engine_public_event_device as base,engine_full_v4_event_device as full,v4_encoder as v4,public_wire
from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex
import test_gpu_engine_pressure_entry_owning as fixtures
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected

class SeismicTossOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.PressureEntryOwningTests.setUpClass();cls.f=fixtures.PressureEntryOwningTests();cls.metadata=deepcopy(cls.f.metadata)
  vocab=cls.metadata['vocabulary'];vocab['abilities']=['_none'];vocab['items']=['_none'];cls.ids={}
  for row in cls.metadata['moves'][1:]:row.update(accuracy=100,priority=0,flags={},source_type='Normal')
  old=cls.f.catalog;moves=dict(old.dex.moves);labels=set(old.vocabulary.tokens)
  for name,label,typ,typid in (('seismictoss','Seismic Toss','Fighting',2),('nightshade','Night Shade','Ghost',8)):
   cls.ids[name]=len(vocab['moves']);vocab['moves'].append(name)
   cls.metadata['moves'].append(dict(id=name,base_power=0,type=typid,category='Physical',source_type=typ,accuracy=100,priority=0,flags={'protect':1,'contact':1},damage='level'))
   moves[name]=replace(old.dex.moves['bodyslam'],id=name,name=label,type=typ,base_power=0,category='Physical',gen3_category='Physical',accuracy=100,priority=0,target='normal',recoil=False,drain=False,heal=False,status=None,boosts={},selfdestruct=False,effect_chance=0,effect_label='')
   labels|={'move:'+name,'belief:possible_move:'+name}
  chart=np.ones((17,17));chart[1,0]=2;chart[1,2]=.5;chart[1,7]=0
  cls.metadata['type_chart']=chart.tolist();dex=ShowdownDex(moves,old.dex.species,old.dex.type_chart)
  cls.catalog=v4.compile_encoder(old.source,dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=old.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.catalog.dex,exposure=fixtures.engine_fixture.EnginePublicEventsTests.exposure)
  projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
  cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog);cls.account=ex.compile_accounting(cls.adapter);cls.hit=direct.compile_direct_hit_controls(engine_catalog=cls.producer)
  cls.context=direct.RegisteredContext(cls.hit.device,cls.account.device);cls.risk=risks.compile_risks(cls.producer,cls.account,direct_hit_compiled=cls.hit)
  cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
 def same(self,a,b):
  for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
 def state(self,b=2):
  s=self.f.state(b);return s._replace(active=j.zeros((b,2),j.int32),entry_pending=j.ones((b,2),bool),done=j.zeros(b,bool),level=j.full_like(s.level,73),hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),stats=j.full_like(s.stats,142).at[:,1,:,5].set(131),ability=j.zeros_like(s.ability),base_ability=j.zeros_like(s.base_ability),item=j.zeros_like(s.item),moves=j.full_like(s.moves,self.ids['seismictoss']),pp=j.full_like(s.pp,24))
 def initial(self,b=2):
  s=self.state(b);cursor=life.register(j.arange(2101,2101+b,dtype=j.int32),**self.kw)
  frame=life.initialize_with_lifecycle(s,self.table,cursor=cursor,**self.kw,tie_coin=j.zeros(b,j.int32),execution_compiled=self.account,kernel_risk_compiled=self.risk)
  consumers=[]
  for owner in (0,1):
   stream=full.register(frame.cursor.stream_id,j.full(b,owner,j.int32),compiled=self.adapter);public=v4.initialize(self.catalog,stream_ids=frame.cursor.stream_id,owner_sides=j.full(b,owner,j.int32))
   result=full.project_and_fold(stream,frame,self.adapter.device,public,self.catalog);self.assertEqual(result.projection.error.tolist(),[0]*b);consumers.append((result.projection.state,result.public))
  return frame.state,frame.cursor,consumers
 def chance(self,b):return fixtures.engine_fixture.EnginePublicEventsTests().chance(b)._replace(draws=j.zeros((b,2,12),j.int32))
 def play(self,state,cursor,*,chance=None,prior=None,context=True):
  b=len(state.active);tape=life._tape(cursor,self.producer,64)._replace(refusal=j.ones(b,j.int32) if prior is None else prior)
  tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=self.account),self.account.device,active=j.ones(b,bool))
  carry=risks.begin_frame_device(risks.initialize_device(tape.stream_id,self.risk.device),tape,self.risk.device,active=j.ones(b,bool));tape=tape._replace(execution=tape.execution._replace(risks=risks.Context(carry,self.risk.device)))
  return tr.step_with_chance(state,j.zeros((b,2),j.int32),self.table,self.chance(b) if chance is None else chance,_public_trace=tape,_lifecycle_trace=True,_direct_hit_context=self.context if context else None)
 def reference_words(self,tape,row,owner):
  names=tuple(self.catalog.species.names[int(self.adapter.device.causal.species[i])] for i in (1,2));ident={side:f'p{side+1}a: '+names[side] for side in (0,1)}
  prefix=[f'|switch|{ident[side]}|{names[side]}, L73, M|'+('1000/1000' if side==owner else '100/100') for side in (0,1)]+['|turn|1']
  wire,_=public_wire.project_lines(public_wire.WireState(),prefix,**v4.wire_kwargs(self.catalog));lines=[]
  for index,w in enumerate(np.array(tape.words[row,:int(tape.count[row])])):
   k=int(w[0])
   if k==events.MOVE:lines.append(f'|move|{ident[int(w[1])]}|Seismic Toss|{ident[int(w[3])]}')
   elif k==events.MAIN_HP:
    side,hp,maximum=map(int,(w[3],w[5],w[6]))
    if side!=owner:hp=min(99,(100*hp+maximum-1)//maximum) if hp<maximum else 100;maximum=100
    lines.append(f'|-damage|{ident[side]}|{hp}/{maximum}')
   elif k==life.TURN:lines.append(f'|turn|{int(tape.event_turn[row,index])}')
   elif k==life.ACTION_END:lines.append('|')
   elif k==life.UPKEEP:lines.append('|upkeep')
   else:self.fail('unrepresented actual raw '+str(k))
  _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog))
  return np.array([e.words() for packet in packets for e in packet.events],np.int32)
 def test_canonical_only_original_compiler_and_default_ast_mutation_controls(self):
  move=self.ids['seismictoss'];night=self.ids['nightshade'];self.assertEqual(int(self.hit.device.seismic_toss_move),move)
  self.assertEqual(int(self.risk.device.selected_risk[move]),0);self.assertTrue(int(self.risk.device.selected_risk[night])&risks.NON_WEATHER)
  closed=risks.compile_risks(self.producer,self.account);self.assertTrue(int(closed.device.selected_risk[move])&risks.NON_WEATHER)
  with self.assertRaises(ValueError):direct.check_compiled(replace(self.hit))
  with self.assertRaises(ValueError):jax.jit(lambda dummy:direct.check_compiled(self.hit))(j.array(0))
  before=ast.parse(subprocess.check_output(['git','show','38626f8a:gpu_engine/transition.py'],text=True));after=ast.parse(Path('gpu_engine/transition.py').read_text());baseline=dump_default_program(before);self.assertEqual(baseline,dump_default_program(after));assert_hp_rng_mutations_detected(self,after,baseline)
 def test_actual_fixed_level_write_preserves_numeric_rng_unused_flags_and_both_full_drains(self):
  state,cursor,consumers=self.initial();state=state._replace(types=state.types.at[1,:,:,0].set(3));actual=self.play(state,cursor)
  plain=tr.step_with_chance(state,j.zeros((2,2),j.int32),self.table,self.chance(2));self.same(actual.step.state,plain.state)
  self.assertEqual(actual.step.state.hp[:,0,0].tolist(),[927,927]);self.assertEqual(actual.step.state.hp[:,1,0].tolist(),[927,927]);self.assertEqual(actual.callbacks.refusal.tolist(),[1,1]);self.assertEqual(actual.callbacks.execution.risks.carry.risk.tolist(),[0,0])
  for row in (0,1):
   raw=np.asarray(actual.callbacks.words[row,:int(actual.callbacks.count[row])]);self.assertFalse(bool(np.isin(raw[:,0],[21,22,23]).any()));self.assertEqual((raw[:,0]==2).sum(),2)
  frame=life.LifecycleResult(actual.step.state,actual.callbacks.cursor,actual.callbacks,life.snapshot(cursor),life.snapshot(actual.callbacks.cursor),j.zeros(2,bool),j.zeros(2,j.int32));closed=[]
  for owner,(stream,public) in enumerate(consumers):
   ledger,result=ex.consume_frame_device(actual.callbacks.execution.ledger,frame,stream,public,active=j.ones(2,bool),tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog);closed.append(ledger)
   self.assertEqual(result.projection.error.tolist(),[0,0]);self.assertTrue(bool(ex.report(ledger,tables=self.account.device).scoped_drained.all()));self.assertFalse(bool(result.projection.B4_qualified.any()))
   for row in (0,1):np.testing.assert_array_equal(self.f.words(result.projection,row),self.reference_words(actual.callbacks,row,owner))
  joined,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()));self.assertEqual(joined.stage[:,:2].tolist(),[[3,3],[3,3]])
 def test_miss_substitute_immune_ko_nightshade_and_prior_bits_remain_unknown(self):
  state,cursor,_=self.initial(8)
  state=state._replace(boosts=state.boosts.at[1,0,0,5].set(-6),substitute_hp=state.substitute_hp.at[2,1,0].set(100),
      types=state.types.at[3,1,0,0].set(8),hp=state.hp.at[4,1,0].set(10),moves=state.moves.at[5,0,0,0].set(self.ids['nightshade']))
  chance=self.chance(8)._replace(draws=self.chance(8).draws.at[1,0,6].set(99));prior=j.ones(8,j.int32).at[6].set(65)
  actual=self.play(state,cursor,chance=chance,prior=prior);plain=tr.step_with_chance(state,j.zeros((8,2),j.int32),self.table,chance);self.same(actual.step.state,plain.state)
  self.assertEqual(int(actual.callbacks.refusal[0]),1);self.assertEqual(int(actual.callbacks.refusal[7]),1)
  self.assertTrue(bool(((actual.callbacks.refusal[1:5]&~1)!=0).all()));self.assertTrue(int(actual.callbacks.refusal[6])&64)
  self.assertTrue(int(actual.callbacks.execution.risks.carry.risk[5])&risks.NON_WEATHER)
  for row in (1,2,3,4):
   ledger=actual.callbacks.execution.ledger;direct_slots=(ledger.site[row]==int(ex.Site.DIRECT_HIT))&(ledger.side[row]==1)
   self.assertTrue(bool(j.any(direct_slots)));self.assertTrue(bool(j.all(j.where(direct_slots,ledger.stage[row]==1,True))))
  # Fixed-source capabilities cannot be copied/replaced at the original host boundary.
  changed=replace(self.hit,device=self.hit.device._replace(seismic_toss_move=j.array(self.ids['nightshade'],j.int32)))
  with self.assertRaises(ValueError):direct.check_compiled(changed)
  w=direct.HitWrite(j.zeros(8,j.int32),j.ones(8,bool),j.ones(8,bool),j.zeros(8,bool),j.ones(8,j.int32),j.ones(8,j.int32),j.zeros(8,bool),j.zeros(8,bool),j.zeros(8,bool),j.full(8,1000,j.int32),j.full(8,927,j.int32),j.full(8,1000,j.int32))
  self.assertFalse(bool(direct.validate_write_device(w,8).any()))
  for key,value in (('hit',j.zeros(8,bool)),('immune',j.ones(8,bool)),('protected',j.ones(8,bool)),('enduring',j.ones(8,bool)),('substitute',j.ones(8,bool)),('power',j.ones(8,j.int32))):
   self.assertFalse(bool(direct.validate_write_device(w._replace(**{key:value}),8,data=self.hit.device,move=j.full(8,self.ids['seismictoss'],j.int32)).any()))
 def test_original_fixed_level_early_return_precedes_annotation_hooks(self):
  node=shutil.which('node')
  if not node:self.skipTest('node unavailable')
  script="const root=process.argv[1];const {Dex}=require(root+'/dist/sim/dex.js');const {BattleActions}=require(root+'/dist/sim/battle-actions.js');const move=Dex.mod('gen3').moves.get('seismictoss');const self=new Proxy({},{get:()=>{throw Error('unused damage/critical/annotation hook read')}});const source=new Proxy({level:73},{get:(x,k)=>{if(k!=='level')throw Error('unexpected source read');return x[k]}});const target={runImmunity:()=>true};const amount=BattleActions.prototype.getDamage.call(self,source,target,move);process.stdout.write(JSON.stringify([move.id,move.basePower,move.damage,move.type,move.category,amount]));"
  root=Path(os.environ.get('POKEZERO_SHOWDOWN_ROOT','/Users/scott/workspace/agents/pokemon-showdown-timing-smoke-f76228a'))
  self.assertEqual(subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),events.SHOWDOWN);self.assertTrue((root/'dist/sim/dex.js').is_file())
  result=json.loads(subprocess.check_output([node,'-e',script,str(root)],text=True));self.assertEqual(result,['seismictoss',0,'level','Fighting','Physical',73])
