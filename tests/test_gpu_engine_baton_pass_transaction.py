"""Two source boundaries + actual original request APIs, no engine turn/game."""
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_baton_pass_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_request_order as ordering,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,state,source_provenance,baton,entry
from gpu_engine import engine_self_boost_callbacks as boosts,private_requests
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_self_boost_input as fixtures
import test_gpu_engine_transition_observer_wiring as static
BASE='b34783d025a54d626404659a716d85fd174898e1'
class BatonTransactionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.SelfBoostInputTests.setUpClass();f=fixtures.SelfBoostInputTests;cls.f=f;cls.metadata=deepcopy(f.metadata);dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);name='batonpass';info=dex.move_info(name);moves[name]=info
  if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
  cls.move=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][cls.move]=dict(id=name,base_power=0,type=1,source_type='Normal',category='Status',pp=40,accuracy=True,priority=0,flags={'metronome':1})
  if 'runaway'not in cls.metadata['vocabulary']['abilities']:cls.metadata['vocabulary']['abilities'].append('runaway')
  cls.ability=cls.metadata['vocabulary']['abilities'].index('runaway');cls.dex=ShowdownDex(moves,f.dex.species,f.dex.type_chart);cls.names=('batonpass','surf','tailglow','agility')
  variants={n:Gen3RandbatSpeciesUniverse(n,80,(Gen3RandbatVariant('baton-'+n,'unit',n,'unit',80,cls.names,'Run Away','Leftovers'),))for n in cls.metadata['vocabulary']['species'][1:]}
  src=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-baton-transaction-unit'),variants)
  labels=set(f.catalog.vocabulary.tokens)|{'move:batonpass','belief:possible_move:batonpass','ability:runaway','belief:possible_ability:runaway'}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(src),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False));cls.rules=request.load_source_rules(showdown_root(),cls.metadata)
  cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(cls.req,cls.catalog);cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(cls.req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)));cls.runtime=driver._checked(cls.prepared)
  cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32));cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))
  cls.consume=staticmethod(jax.jit(lambda l,f,s,p:ex.consume_frame_device(l,f,s,p,active=j.ones(len(l.stream_id),bool),tables=cls.runtime['execution_compiled'].device,adapter_tables=cls.runtime['public'].device,catalog=cls.catalog)))
 def initial(self):
  b=2;z=j.zeros(b,j.int32);yes=j.ones(b,bool);streams=j.array([199001,199002],j.int32);indices=np.tile([[0,1]],(b,1))
  species=np.ones((2,6),np.int32);species[:,3]=2;moves=np.broadcast_to(np.array([self.metadata['vocabulary']['moves'].index(n)for n in self.names],np.int32),(2,6,4)).copy();pp=np.broadcast_to(np.array([64,24,32,48],np.int32),(2,6,4)).copy();types=np.asarray(self.rules.device.species_types)[species]
  arrays=dict(species=species,level=np.full((2,6),80,np.int32),types=types,stats=np.full((2,6,6),100,np.int32),moves=moves,pp=pp,item=np.zeros((2,6),np.int32),ability=np.full((2,6),self.ability,np.int32),hidden_power_type=np.full((2,6),7,np.int32),hidden_power_power=np.full((2,6),70,np.int32),gender=np.ones((2,6),np.int32),nature=np.zeros((2,6),np.int32),evs=np.zeros((2,6,6),np.int32),metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8))
  with TemporaryDirectory()as tmp:
   p=Path(tmp)/'unit.npz';np.savez_compressed(p,**arrays);pool=source_provenance.load_constructor_pool(p)
  # Literal typed source fixture: no generator, constructor physics or RNG call.
  s=state.empty_state(b)._replace(species=j.asarray(species[indices]),level=j.asarray(arrays['level'][indices]),types=j.asarray(types[indices]),base_types=j.asarray(types[indices]),stats=j.asarray(arrays['stats'][indices]),max_hp=j.full((b,2,6),100,j.int32),hp=j.full((b,2,6),100,j.int32),moves=j.asarray(moves[indices]),pp=j.asarray(pp[indices]),ability=j.full((b,2,6),self.ability,j.int32),base_ability=j.full((b,2,6),self.ability,j.int32),gender=j.ones((b,2,6),j.int32),hidden_power_type=j.full((b,2,6),7,j.int32),hidden_power_power=j.full((b,2,6),70,j.int32),entry_pending=j.zeros((b,2),bool),done=~yes)
  regs=tuple(request._registration(s,pool,indices,streams,z+side)for side in(0,1));orders=tuple(ordering.initialize_order(streams,z+side,z+6,engine_binding=j.broadcast_to(self.rules.device.engine_binding,(b,8)))for side in(0,1));cursor=life.register(streams,**self.kw);initial=life._tape(cursor,self.producer,64)
  for side in(0,1):initial=life._reveal(initial,s,z+side,z-1,z,eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,z);owners=[]
  for owner in(0,1):
   stream=full.register(streams,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=streams,owner_sides=z+owner);out=self.fold(stream,frame,pub);self.assertEqual(out.projection.error.tolist(),[0,0]);owners.append((out.projection.state,out.public))
  return SimpleNamespace(s=s,cursor=initial.cursor,owners=owners,regs=regs,orders=orders)
 def new_tape(self,x,ledger=None):
  b=len(x.s.hp);yes=j.ones(b,bool);t=life._tape(x.cursor,self.producer,64);a=self.runtime['execution_compiled'];r=self.runtime['kernel_risk_compiled'];l=ex.initialize(x.cursor.stream_id,compiled=a)if ledger is None else ledger;t=ex.begin_on_tape(t,l,a.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(t.stream_id,r.device)._replace(frame=l.frame),t,r.device,active=yes);return t._replace(execution=t.execution._replace(risks=risk.Context(carry,r.device)))
 def produced(self,s,orders,regs,cursor):return tuple(request.produce_request(self.req,s,o,reg,life.snapshot(cursor),lifecycle_admitted=j.ones(len(s.hp),bool))for o,reg in zip(orders,regs))
 def prepare_tape(self,x,t,history,requests,actions):
  proof=cb.public_proof(x.s,x.cursor,tuple(p for _,p in x.owners),j.ones(2,bool),self.helper.device.baton_pass)
  return cb.prepare(t,x.s,history,proof,requests,x.orders,actions,self.helper.device.baton_pass)
 def drain(self,x,t,s,tag):
  z=j.zeros(2,j.int32);yes=j.ones(2,bool);frame=life.LifecycleResult(s,t.cursor,t,life.snapshot(x.cursor),life.snapshot(t.cursor),~yes,z);new=[];ledgers=[];actual=[]
  for owner,(stream,pub)in enumerate(x.owners):
   ledger,r=self.consume(t.execution.ledger,frame,stream,pub);actual.append(dict(owner=owner,error=r.projection.error.tolist(),words=r.projection.frames.words.tolist(),valid=r.projection.frames.valid.tolist(),stage=ledger.stage.tolist(),refusal=ledger.refusal.tolist()))
   Path('/tmp/pokezero-baton-'+tag+'-actual.json').write_text(json.dumps(actual)+'\n')
   self.assertEqual(r.projection.error.tolist(),[0,0],actual[-1]);new.append((r.projection.state,r.public));ledgers.append(ledger)
  return frame,new,ledgers
 def start_component(self):
  x=self.initial();b=2;r=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);side=r%2
  # A genuine original typed TailGlow write supplies the nonzero before-public
  # boosts. It is not an unobserved final-state or a chosen acceptance flag.
  requests=self.produced(x.s,x.orders,x.regs,x.cursor);history=cb.empty_history(x.s,x.cursor,j.stack(tuple(o.request_to_physical for o in x.orders),axis=1));t=self.prepare_tape(x,self.new_tape(x),history,requests,j.zeros((b,2),j.int32));t=life.begin(t,x.s,yes,~yes)
  move=z+self.metadata['vocabulary']['moves'].index('tailglow');t=public.move(t,side,z,z,move,eligible=yes);old=x.s.boosts[r,side,0];t,p=boosts.begin(t,x.s,side,move,move,old,self.runtime['direct_hit_context'],self.table.effects,eligible=yes);updated=j.clip(old+self.table.effects.self_boosts[move],-6,6);t,c=boosts.finish(t,p,updated,yes)
  s=x.s._replace(boosts=x.s.boosts.at[r,side,0].set(updated));_,owners,ledgers=self.drain(x,t,s,'prior-boost');x.s=s;x.cursor=t.cursor;x.owners=owners;ledger=ex.join_consumed_owners(ledgers[0],ledgers[1],tables=self.runtime['execution_compiled'].device)[0]
  # Same original PP source, captured after the ordinary declaration spends1.
  x.s=x.s._replace(pp=x.s.pp.at[r,side,0,0].add(-1));requests=self.produced(x.s,x.orders,x.regs,x.cursor);history=cb.empty_history(x.s,x.cursor,j.stack(tuple(o.request_to_physical for o in x.orders),axis=1));t=self.prepare_tape(x,self.new_tape(x,ledger),history,requests,j.zeros((b,2),j.int32));t=life.begin(t,x.s,yes,~yes);move=z+self.move;t=public.move(t,side,z,z,move,eligible=yes)
  before=t;t,p=cb.begin_start(t,x.s,side,move,move,self.runtime['direct_hit_context'],acts=yes,eligible=yes);bench=j.any((x.s.hp[r,side]>0)&(j.arange(6)[None,:]!=x.s.active[r,side,None]),axis=1)
  step=static._step(Path('gpu_engine/transition.py').read_text());assignment=next(n for n in ast.walk(step)if isinstance(n,ast.Assign)and any(isinstance(a,ast.Name)and a.id=='passed'for a in n.targets));passed=eval(compile(ast.Expression(assignment.value),'actual-Baton-pass','eval'),{},dict(acts=yes,table=self.table,move=move,hp=x.s.hp[r,side,0],bench_alive=bench))
  s=x.s._replace(baton_pass=x.s.baton_pass.at[r,side].set(passed));t=cb.finish_start(t,p,s,passed,yes);s=s._replace(forced_switch=s.baton_pass,pending_residual=yes,turn=z+1);t=cb.finalize_request(t,s);self.assertEqual(cb.owned(t,t.count-1,cb.START).tolist(),[True,True]);return SimpleNamespace(x=x,s=s,t=t,side=side,before=before,pending=p)
 def test_two_boundaries_original_both_full_and_private_request(self):
  c=self.start_component();x=c.x;b=2;r=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool)
  frame,owners,ledgers=self.drain(x,c.t,c.s,'start');requests=self.produced(c.s,x.orders,x.regs,c.t.cursor)
  for owner,q in enumerate(requests):self.assertEqual(q.actions.facts.request.kind.tolist(),[int(private_requests.RequestKind.FORCE_SWITCH)if int(side)==owner else int(private_requests.RequestKind.WAIT)for side in c.side])
  history=cb.mature(c.t.execution.baton_transaction.history,frame,tuple(ledgers),tuple(p for _,p in owners),requests);self.assertEqual(history.ready.tolist(),[True,True]);x.s=c.s;x.cursor=c.t.cursor;x.owners=owners
  ledger=ex.join_consumed_owners(ledgers[0],ledgers[1],tables=self.runtime['execution_compiled'].device)[0];actions=j.zeros((b,2),j.int32).at[r,c.side].set(7);t=self.prepare_tape(x,self.new_tape(x,ledger),history,requests,actions);self.assertEqual(t.execution.baton_transaction.request_ok.tolist(),[True,True]);t=life.begin(t,x.s,yes,~yes)
  target=z+3
  # Only negative mutations of the genuine consumed source/request transaction.
  trn=t.execution.baton_transaction
  for name,h in [('stale',history._replace(frame=history.frame-1)),('foreign',history._replace(stream_id=history.stream_id+1)),('unconsumed',history._replace(ready=j.zeros(b,bool)))]:
   bad=self.prepare_tape(x,self.new_tape(x,ledger),h,requests,actions);self.assertFalse(bool(bad.execution.baton_transaction.request_ok.any()),name)
  bad_proof=trn.public._replace(pending=j.zeros_like(trn.public.pending));bad=cb.prepare(self.new_tape(x,ledger),x.s,history,bad_proof,requests,x.orders,actions,self.helper.device.baton_pass);self.assertFalse(bool(bad.execution.baton_transaction.request_ok.any()))
  swapped=x.orders[0]._replace(request_to_physical=x.orders[0].request_to_physical.at[:,1].set(2).at[:,2].set(1));bad=cb.prepare(self.new_tape(x,ledger),x.s,history,trn.public,requests,(swapped,x.orders[1]),actions,self.helper.device.baton_pass);self.assertFalse(bool(bad.execution.baton_transaction.request_ok.any()))
  _,wrong=cb.before_copy(t,x.s,c.side,target+1,self.runtime['direct_hit_context'],eligible=yes);self.assertFalse(bool(wrong.valid.any()))
  t,p=cb.before_copy(t,x.s,c.side,target,self.runtime['direct_hit_context'],eligible=yes);self.assertEqual(p.valid.tolist(),[True,True]);after=baton.copy_to_recipient(x.s,c.side,target,eligible=yes);t=cb.after_copy(t,p,after);after=entry.depart(after,c.side,self.table.entry,eligible=yes);after=after._replace(active=after.active.at[r,c.side].set(target),baton_pass=after.baton_pass.at[r,c.side].set(False),forced_switch=after.forced_switch.at[r,c.side].set(False));plain=after
  self.assertEqual(cb.reveal_matches(t,after,c.side,z,target).tolist(),[True,True])
  t=life.installed(t,after,c.side,z,target,eligible=yes,replacement=yes,baton=yes);t=cb.finish_install(t,after,c.side,eligible=yes)
  Path('/tmp/pokezero-baton-install-input.json').write_text(json.dumps(dict(raw=t.words.tolist(),count=t.count.tolist(),refusal=t.refusal.tolist(),stage=t.execution.ledger.stage.tolist(),record_known=t.execution.baton_transaction.record.known.tolist(),finalized=t.execution.baton_transaction.record.finalized.tolist()))+'\n');self.assertEqual(cb.owned(t,t.count-1,cb.INSTALL).tolist(),[True,True])
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))});self.assertFalse(bool(cb.owned(bad,t.count-1,cb.INSTALL).any()),name)
   with self.assertRaises(ValueError):cb.owned(t._replace(**{name:j.broadcast_to(getattr(t,name),(8,8))}),t.count-1,cb.INSTALL)
  bad=t._replace(count=t.count-1);self.assertFalse(bool(cb.owned(bad,t.count-2,cb.INSTALL).any()))
  rec=t.execution.baton_transaction.record;bad_rec=rec._replace(boost_after=rec.boost_after.at[0,0,3,2].add(1));bad=t._replace(execution=t.execution._replace(baton_transaction=t.execution.baton_transaction._replace(record=bad_rec)));self.assertEqual(cb.owned(bad,t.count-1,cb.INSTALL).tolist(),[False,True])
  absent=x.owners[0][1]._replace(context=x.owners[0][1].context._replace(tendency=x.owners[0][1].context.tendency._replace(baton=j.zeros((b,2),bool))));self.assertFalse(bool(cb.public_guard(t,absent).any()))
  install_frame,new,closed=self.drain(x,t,after,'install');self.assertEqual(cb.public_guard(t,x.owners[0][1]).tolist(),[True,True]);expected=[]
  for owner,((oldstream,oldpub),(stream,pub))in enumerate(zip(x.owners,new)):
   for row,side in enumerate(c.side.tolist()):
    initial=self.catalog.species.names[1];incoming=self.catalog.species.names[2];ids={s:f'p{s+1}a: '+initial for s in(0,1)};target_ident=f'p{side+1}a: '+incoming
    prefix=[f'|switch|{ids[s]}|{initial}, L80, M|100/100'for s in(0,1)]+['|turn|1','|','|t:|1',f'|move|{ids[side]}|Tail Glow|{ids[side]}',f'|-boost|{ids[side]}|spa|2']
    wire,_=public_wire.project_lines(public_wire.WireState(),prefix,**v4.wire_kwargs(self.catalog));wire,start=public_wire.project_lines(wire,['|','|t:|1',f'|move|{ids[side]}|Baton Pass|{ids[side]}'],**v4.wire_kwargs(self.catalog));_,out=public_wire.project_lines(wire,['|','|t:|1',f'|switch|{target_ident}|{incoming}, L80, M|100/100'],**v4.wire_kwargs(self.catalog));want=np.array([e.words()for p in out for e in p.events],np.int32)
    actual_doc=json.loads(Path('/tmp/pokezero-baton-install-actual.json').read_text())[owner];words=np.array(actual_doc['words'],np.int32);valid=np.array(actual_doc['valid']);actual=words[:,row][valid[:,row]];expected.append(dict(owner=owner,row=row,actual=actual.tolist(),expected=want.tolist()));Path('/tmp/pokezero-baton-original-both-full-compact.json').write_text(json.dumps(expected)+'\n');np.testing.assert_array_equal(actual,want)
    self.assertEqual(int(pub.mons.boosts[row,side,2]),2);self.assertTrue(bool(pub.mons.arrived_by_baton[row,side]));self.assertFalse(bool(pub.context.tendency.baton[row,side]))
  orders=tuple(life.apply_installations(o,install_frame)for o in x.orders);current=self.produced(after,orders,x.regs,t.cursor)
  for owner,q in enumerate(current):
   self.assertEqual(q.actions.admitted.tolist(),[True,True]);self.assertEqual(q.actions.boundary.generation.tolist(),orders[owner].generation.tolist());self.assertEqual(q.actions.facts.request.pp[:,0].tolist(),[64 if int(side)==owner else 64 for side in c.side])
  for a,z in zip(jax.tree.leaves(after),jax.tree.leaves(plain),strict=True):np.testing.assert_array_equal(a,z)
 def test_static_original_source_default_legacy_and_imports(self):
  self.assertEqual(int(self.helper.device.baton_pass.move_id),self.move);self.assertTrue(bool(self.helper.device.baton_pass.quiet_abilities[self.ability]));self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.baton_pass)
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(subprocess.check_output(['git','show','567eb6c0:gpu_engine/transition.py'],text=True));baseline=dump_default_program(old);self.assertEqual(dump_default_program(new),baseline);assert_hp_rng_mutations_detected(self,new,baseline);self.assertEqual(static._undefined_step_globals(Path('gpu_engine/transition.py').read_text()),set())
  current=ast.parse(Path('gpu_engine/transition.py').read_text());integrated=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True))
  self.assertEqual(ast.dump(current),ast.dump(integrated));assert_hp_rng_mutations_detected(self,current,dump_default_program(integrated))
  for file in('baton.py','entry.py','state.py'):
   self.assertEqual(subprocess.check_output(['git','show','567eb6c0:gpu_engine/'+file]),subprocess.check_output(['git','show',BASE+':gpu_engine/'+file]))
   self.assertEqual(Path('gpu_engine',file).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+file]))
  script="""const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),P=require(root+'/dist/sim/pokemon.js').Pokemon,m=d.moves.get('batonpass'),s={side:{},volatiles:{},boosts:{atk:1,def:0,spa:2,spd:0,spe:3,accuracy:0,evasion:0},clearVolatile(){this.boosts={atk:0,def:0,spa:0,spd:0,spe:0,accuracy:0,evasion:0}}},t={volatiles:{},clearVolatile(){this.volatiles={}},battle:{}};m.onHit.call({canSwitch:()=>true},s);m.self.onHit.call({},s);P.prototype.copyVolatileFrom.call(t,s,'copyvolatile');console.log(JSON.stringify({skip:s.skipBeforeSwitchOutEventFlag,old:s.boosts,new:t.boosts}));"""
  v=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertTrue(v['skip']);self.assertEqual(v['new']['spa'],2);self.assertTrue(all(n==0 for n in v['old'].values()))
  import os,sys
  for names in [('gpu_engine.benchmark_retained_games','gpu_engine.engine_baton_pass_callbacks','gpu_engine.engine_full_v4_event_device'),('gpu_engine.engine_full_v4_event_device','gpu_engine.engine_baton_pass_callbacks','gpu_engine.benchmark_retained_games')]:
   out=subprocess.run([sys.executable,'-B','-c','import importlib;'+''.join('importlib.import_module('+repr(n)+');'for n in names)],capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),timeout=30);self.assertEqual(out.returncode,0,out.stderr)
 def test_final_registered_context_and_closed_mode_binding(self):
  x=self.initial();yes=j.ones(2,bool);z=j.zeros(2,j.int32);q=self.produced(x.s,x.orders,x.regs,x.cursor);h=cb.empty_history(x.s,x.cursor,j.stack(tuple(o.request_to_physical for o in x.orders),axis=1));t=self.prepare_tape(x,self.new_tape(x),h,q,j.zeros((2,2),j.int32));t=life.begin(t,x.s,yes,~yes);side=j.arange(2)%2;move=z+self.move;t=public.move(t,side,z,z,move,eligible=yes)
  _,p=cb.begin_start(t,x.s,side,move,move,self.runtime['direct_hit_context'],acts=yes,eligible=yes);self.assertEqual(p.valid.tolist(),[True,True])
  with self.assertRaises(ValueError):cb.begin_start(t,x.s,side,move,move,direct.Context(self.helper.device),acts=yes,eligible=yes)
  _,bad=cb.begin_start(t,x.s,side,move,move+1,self.runtime['direct_hit_context'],acts=yes,eligible=yes);self.assertFalse(bool(bad.valid.any()))
  source=Path('gpu_engine/engine_retained_encoder_driver.py').read_text();tree=ast.parse(source);frame=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_frame')
  condition=next(n.test for n in ast.walk(frame)if isinstance(n,ast.If)and any(isinstance(c,ast.Call)and isinstance(c.func,ast.Attribute)and c.func.attr=='prepare'and isinstance(c.func.value,ast.Name)and c.func.value.id=='baton_callbacks'for c in ast.walk(n)))
  text=ast.unparse(condition);self.assertIn('RegisteredContext',text);self.assertIn('execution_tables is not None',text);self.assertIn('kernel_risk_tables is not None',text)
  self.assertEqual(source.count("if r['direct_hit_mode']=='kernel-event' and r['execution_compiled'] is not None and r['direct_hit_context'] is not None and r['direct_hit_context'].data.baton_pass is not None:"),2)
 def test_missing_owning_public_transaction_is_entered_unknown(self):
  x=self.initial();yes=j.ones(2,bool);z=j.zeros(2,j.int32);side=j.arange(2)%2;move=z+self.move;t=self.new_tape(x);t=life.begin(t,x.s,yes,~yes);t=public.move(t,side,z,z,move,eligible=yes)
  before=t.count;t,p=cb.begin_start(t,x.s,side,move,move,self.runtime['direct_hit_context'],acts=yes,eligible=yes);self.assertIsNone(p);self.assertTrue(bool(j.all((t.refusal&256)!=0)));self.assertTrue(bool(j.all(t.execution.ledger.site[:,1]==39)));self.assertTrue(bool(j.all(t.execution.ledger.stage[:,1]==1)));np.testing.assert_array_equal(t.count,before)
  s=x.s._replace(baton_pass=x.s.baton_pass.at[j.arange(2),side].set(True));t=self.new_tape(x);t=life.begin(t,s,yes,~yes);t,p=cb.before_copy(t,s,side,z+3,self.runtime['direct_hit_context'],eligible=yes);self.assertIsNone(p);self.assertTrue(bool(j.all((t.execution.ledger.refusal&ex.UNKNOWN_SITE)!=0)))
 def test_private_fixture_requests(self):
  x=self.initial();requests=self.produced(x.s,x.orders,x.regs,x.cursor);self.assertEqual([q.actions.admitted.tolist()for q in requests],[[True,True],[True,True]])
class BatonCallerBindingsTests(unittest.TestCase):
 def test_exact_saved_suspension_copy_and_current_state_join(self):
  s=state.empty_state(2);saved=cb.suspend(s);self.assertTrue(bool(cb.equal(saved,cb.suspend(s)).all()))
  self.assertEqual(tuple(saved._fields),cb.SUSPENSION_FIELDS)
  for name in cb.SUSPENSION_FIELDS:
   current=getattr(s,name);changed=~current if current.dtype==j.bool_ else current+1
   bad=s._replace(**{name:changed});self.assertFalse(bool(cb.equal(saved,cb.suspend(bad)).any()),name)
   np.testing.assert_array_equal(getattr(saved,name),current)
  cb.suspension_shape(saved,2)
  for name in cb.SUSPENSION_FIELDS:
   value=getattr(saved,name)
   for bad in (value[...,None],value.astype(j.float32)):
    with self.assertRaises(ValueError,msg=name):cb.suspension_shape(saved._replace(**{name:bad}),2)
  tree=ast.parse(Path('gpu_engine/engine_baton_pass_callbacks.py').read_text());prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='prepare');joined=[n for n in ast.walk(prepare)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='equal'and ast.unparse(n.args[0])=='h.suspended'];self.assertEqual(len(joined),1);self.assertEqual(ast.unparse(joined[0].args[1]),'suspend(state)')
  final=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='finalize_request');self.assertIn('suspended=suspend(state)',ast.unparse(final))
 def test_maturation_joins_the_exact_consumed_history(self):
  from unittest.mock import patch
  s=state.empty_state(2);z=j.zeros(2,j.int32);cursor=SimpleNamespace(stream_id=z+9100,turn=z,generation=j.zeros((2,2),j.int32));h=cb.empty_history(s,cursor,j.broadcast_to(j.arange(6),(2,2,6)))._replace(side=z,frame=z,public_slot=z)
  rec=cb.empty_record(h)._replace(code=z,slot=z,serial=z+1);trn=cb.Transaction(h,None,None,None,None,rec);frame=SimpleNamespace(callbacks=SimpleNamespace(execution=SimpleNamespace(baton_transaction=trn)))
  ledger=SimpleNamespace(site=j.full((2,1),39,j.int32),stream_id=h.stream_id,refusal=z,stage=j.full((2,1),3,j.int32),serial=j.ones((2,1),j.int32));pub=SimpleNamespace(cursor=SimpleNamespace(stream_id=h.stream_id),context=SimpleNamespace(tendency=SimpleNamespace(baton=j.ones((2,2),bool))),mons=SimpleNamespace(boosts=j.zeros((2,2,7),j.int32)))
  from gpu_engine import public_active
  pub.context.active=SimpleNamespace(volatiles=j.zeros((2,2,len(public_active.NAMES)),bool))
  def request(kind):return SimpleNamespace(actions=SimpleNamespace(admitted=j.ones(2,bool),facts=SimpleNamespace(request=SimpleNamespace(kind=j.full(2,int(kind),j.int32)))))
  requests=(request(private_requests.RequestKind.FORCE_SWITCH),request(private_requests.RequestKind.WAIT))
  with patch.object(cb,'owned',return_value=j.ones(2,bool)):
   self.assertEqual(cb.mature(h,frame,(ledger,ledger),(pub,pub),requests).ready.tolist(),[True,True])
   for altered in (h._replace(private=h.private._replace(hp=h.private.hp.at[0,0,0].add(1))),h._replace(suspended=h.suspended._replace(queued_actor=h.suspended.queued_actor.at[0].add(1)))):
    self.assertEqual(cb.mature(altered,frame,(ledger,ledger),(pub,pub),requests).ready.tolist(),[False,True])
   changed=h._replace(order=h.order.at[0,0,1].set(5));other=frame.callbacks.execution.baton_transaction._replace(history=changed);badframe=SimpleNamespace(callbacks=SimpleNamespace(execution=SimpleNamespace(baton_transaction=other)))
   self.assertEqual(cb.mature(h,badframe,(ledger,ledger),(pub,pub),requests).ready.tolist(),[False,True])
 def test_actual_production_observer_operands(self):
  source=Path('gpu_engine/transition.py').read_text();step=static._step(source);self.assertEqual(static._undefined_step_globals(source),set());sentinel=object();side=j.array([0,1],j.int32);rows=j.arange(2);target=j.array([[3,0],[0,3]],j.int32);yes=j.ones(2,bool)
  names=('before_copy','after_copy','finish_install','begin_start','finish_start','finalize_request');spy=SimpleNamespace(**{n:(lambda *a,**kw:(a,kw))for n in names});env=dict(baton_callbacks=spy,_public_trace=sentinel,work=sentinel,side=side,target=target,rows=rows,_direct_hit_context=sentinel,_direct_hit_registered=True,do_switch=yes,_baton_copy=sentinel,_baton_start=sentinel,move=sentinel,caller_move=sentinel,acts=yes,caller_acts=yes,passed=yes,hit=yes,immune=~yes)
  calls={n.func.attr:n for n in ast.walk(step)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='baton_callbacks'}
  self.assertEqual(set(calls),set(names))
  for name,call in calls.items():
   args,kw=eval(compile(ast.Expression(call),'actual-Baton-observer','eval'),{},env);self.assertIs(args[0],sentinel);self.assertIs(args[1],sentinel)
   if name=='before_copy':np.testing.assert_array_equal(args[3],[3,3]);self.assertIs(args[4],sentinel);self.assertIs(kw['eligible'],yes)
   elif name=='begin_start':self.assertIs(args[3],sentinel);self.assertIs(args[4],sentinel);self.assertIs(args[5],sentinel);self.assertIs(kw['eligible'],yes)
   elif name=='finish_install':self.assertIs(args[2],side);self.assertIs(kw['eligible'],yes)
   elif name=='finish_start':self.assertIs(args[3],yes);np.testing.assert_array_equal(args[4],yes)
  driver_source=Path('gpu_engine/engine_retained_encoder_driver.py').read_text();self.assertIn('baton_transaction=None if tape.execution.baton_transaction is None else baton_callbacks.rows',driver_source);self.assertIn('baton_public=baton_public',driver_source);self.assertIn('baton_history=core.baton_history if frame.callbacks.execution is None',driver_source);self.assertIn('baton_callbacks.mature(',driver_source)

if __name__=='__main__':unittest.main()
