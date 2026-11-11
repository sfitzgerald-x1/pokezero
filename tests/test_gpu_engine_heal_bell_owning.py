"""Original callback/write-mask -> BOTH original FULL, no kernel/model/game."""
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_heal_bell_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public,engine_request_order as order
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,state,effects
from gpu_engine import engine_source_frame_witness as witness
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixtures
BASE='afc80dc70525b7eb8723233015ce6495614bef08'

class HealBellTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.NoDamageOwningTests.setUpClass();f=fixtures.NoDamageOwningTests;cls.metadata=deepcopy(f.metadata);dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves)
  # Unused legacy Hidden Power fixture rows are outside this component and
  # predate the stricter member-source compiler; omit that unrelated namespace.
  kept=[i for i,n in enumerate(cls.metadata['vocabulary']['moves'])if not n.startswith('hiddenpower')]
  cls.metadata['vocabulary']['moves']=[cls.metadata['vocabulary']['moves'][i]for i in kept];cls.metadata['moves']=[cls.metadata['moves'][i]for i in kept]
  for name in ('rest','sleeptalk','wish','healbell','surf'):
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   i=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=100 if name=='surf'else True,priority=info.priority,flags={'protect':1}if name=='surf'else {'snatch':1,'sound':1,'metronome':1}if name=='healbell'else{})
  for name in ('soundproof',):
   if name not in cls.metadata['vocabulary']['abilities']:cls.metadata['vocabulary']['abilities'].append(name)
  if 'encore'in cls.metadata['vocabulary']['moves']:
   info=dex.move_info('encore');moves['encore']=info;cls.metadata['moves'][cls.metadata['vocabulary']['moves'].index('encore')]=dict(id='encore',base_power=0,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=100,priority=0,flags={'protect':1,'mirror':1,'bypasssub':1,'metronome':1,'failencore':1})
  cls.dex=ShowdownDex(moves,f.dex.species,f.dex.type_chart);names=('healbell','surf','rest','sleeptalk')
  variants={name:Gen3RandbatSpeciesUniverse(name,80,tuple(Gen3RandbatVariant('heal-'+name+ability,'authored',name,'authored',80,names,ability,'Leftovers')for ability in ('Pressure','Soundproof')))for name in cls.metadata['vocabulary']['species'][1:]}
  src=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-heal-bell'),variants)
  labels=set(f.catalog.vocabulary.tokens)
  for name in ('rest','sleeptalk','wish','healbell','surf'):labels|={'move:'+name,'belief:possible_move:'+name}
  labels|={'ability:soundproof','belief:possible_ability:soundproof'}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(src),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__)
  original_flags=json.loads(subprocess.check_output(['node','-e',"const d=require(process.argv[1]).Dex.mod('gen3');console.log(JSON.stringify(JSON.parse(process.argv[2]).map(n=>!!d.moves.get(n).flags.failencore)))",str(showdown_root()/'dist/sim/dex.js'),json.dumps(cls.metadata['vocabulary']['moves'])],text=True))
  for row,flag in zip(cls.metadata['moves'],original_flags):
   if row is not None:row.setdefault('flags',{})['failencore']=flag
  for family,name in (('abilities','insomnia'),('items','lumberry')):
   if name not in cls.metadata['vocabulary'][family]:cls.metadata['vocabulary'][family].append(name)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))

 def component(self,consume_initial=True):
  b=8;rows=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);actor=rows%2
  species=j.ones((b,2,6),j.int32).at[:,:,0].set(2)
  hp=j.full((b,2,6),100,j.int32);status=j.zeros((b,2,6),j.int32).at[:,:,0].set(2).at[:,:,1].set(1)
  ability=j.zeros((b,2,6),j.int32);sp=self.metadata['vocabulary']['abilities'].index('soundproof')
  ability=ability.at[1,actor[1],1].set(sp).at[2,actor[2],0].set(sp).at[5,actor[5],:].set(sp)
  # Healthy/no-write, dead skip, and all six member/provenance facts coexist.
  status=status.at[3,actor[3],:].set(0);hp=hp.at[4,actor[4],2].set(0);status=status.at[4,actor[4],2].set(3)
  s=state.empty_state(b)._replace(active=j.zeros((b,2),j.int32),species=species,hp=hp,max_hp=j.full((b,2,6),100,j.int32),status=status,status_counter=j.where(status!=0,2,0),sleep_skipped=j.where(status==5,1,0),rest_origin=j.zeros((b,2,6),j.int32),ability=ability,gender=j.ones((b,2,6),j.int32),level=j.full((b,2,6),80,j.int32))
  kw=dict(catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));cursor=life.register(rows+97001,**kw);initial=life._tape(cursor,self.producer,64)
  for side in(0,1):initial=life._reveal(initial,s,z+side,z-1,z,eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,z);consumers=[]
  for owner in((0,1)if consume_initial else()):
   stream=full.register(cursor.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=z+owner);r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error.tolist(),[0]*b);consumers.append((r.projection.state,r.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rt),tape,rt,active=yes)
  orders=tuple(order.initialize_order(cursor.stream_id,z+side,z+6,engine_binding=j.broadcast_to(cursor.engine_binding,(b,8)))for side in(0,1))
  tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rt),party_orders=orders))
  s=s._replace(active=j.ones((b,2),j.int32))
  for side in(0,1):tape=life._reveal(tape,s,z+side,z,z+1,eligible=yes,replacement=~yes,baton=~yes,initial=False)
  return s,tape,consumers,actor

 def test_source_same_mask_order_and_closed_negatives(self):
  import symtable,builtins
  text=Path('gpu_engine/transition.py').read_text();symbols=symtable.symtable(text,'actual-transition','exec');module_names={v.get_name()for v in symbols.get_symbols()if v.is_assigned()or v.is_imported()}|set(dir(builtins))|{'__name__','__file__'}
  fn=next(v for v in symbols.get_children()if v.get_name()=='step_with_chance');pending=[fn];undefined=set()
  while pending:
   scope=pending.pop();pending.extend(scope.get_children())
   undefined.update(v.get_name()for v in scope.get_symbols()if v.is_referenced()and v.is_global()and v.get_name()not in module_names)
  self.assertEqual(undefined,set())
  actual=ast.parse(text);step=next(n for n in actual.body if isinstance(n,ast.FunctionDef)and n.name=='step_with_chance')
  calls=[n for n in ast.walk(step)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='heal_bell_callbacks']
  self.assertEqual({n.func.attr for n in calls},{'begin','prospective','finish'})
  begin=next(n for n in calls if n.func.attr=='begin');finish=next(n for n in calls if n.func.attr=='finish')
  self.assertEqual([k.arg for k in begin.keywords],['eligible','primary','secondary']);self.assertEqual([ast.unparse(v)for v in finish.args],['_public_trace','_heal_bell_pending','work','_heal_bell_mask'])
  capture=[n for n in ast.walk(step)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Tuple)and any(isinstance(v,ast.Name)and v.id=='_heal_bell_mask'for v in t.elts)for t in n.targets)]
  self.assertEqual(len(capture),1);self.assertTrue(next(k.value.value for k in capture[0].value.keywords if k.arg=='_return_write_mask'))
  witness._source();self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.heal_bell)
  self.assertEqual(int(self.helper.device.heal_bell.move),self.metadata['vocabulary']['moves'].index('healbell'))
  # The numerical cure body is exact; the added return exposes its SAME mask.
  effect_old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/effects.py'],text=True));effect_new=ast.parse(Path('gpu_engine/effects.py').read_text());get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='cure_team')
  old_fn=get(effect_old);new_fn=get(effect_new);self.assertEqual(new_fn.args.kwonlyargs[-1].arg,'_return_write_mask');new_fn.args.kwonlyargs.pop();new_fn.args.kw_defaults.pop()
  self.assertIsInstance(new_fn.body[-2],ast.Assign);self.assertIsInstance(new_fn.body[-1].value,ast.IfExp);new_fn.body[-2:]=[ast.Return(value=new_fn.body[-2].value)]
  self.assertEqual(ast.dump(old_fn,include_attributes=False),ast.dump(new_fn,include_attributes=False))
  # Default transition program including original HP/RNG expressions is exact.
  old=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True));new=ast.parse(Path('gpu_engine/transition.py').read_text());baseline=dump_default_program(old);self.assertEqual(dump_default_program(new),baseline);assert_hp_rng_mutations_detected(self,new,baseline)
  s,t,_,actor=self.component(consume_initial=False);b=8;rows=j.arange(b);move=j.full(b,int(self.helper.device.heal_bell.move),j.int32);yes=j.ones(b,bool)
  t=public.move(t,actor,s.active[rows,actor],s.active[rows,1-actor],move,eligible=yes);t,p=cb.begin(t,s,actor,move,self.runtime['direct_hit_context'],self.table,eligible=yes,primary=yes,secondary=yes)
  np.testing.assert_array_equal(p.permutation,np.tile([1,0,2,3,4,5],(b,1)))
  after,mask=effects.cure_team(s,actor,self.table.effects,eligible=yes,_return_write_mask=True);plain=effects.cure_team(s,actor,self.table.effects,eligible=yes)
  for x,y in zip(jax.tree.leaves(after),jax.tree.leaves(plain)):np.testing.assert_array_equal(x,y)
  done,c=cb.finish(t,p,after,mask);self.assertTrue(bool(cb.captured_matches(done,c).all()))
  for field in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=t._replace(**{field:getattr(t,field).at[7].add(j.uint32(1))});_,p2=cb.begin(bad,s,actor,move,self.runtime['direct_hit_context'],self.table,eligible=yes,primary=yes,secondary=yes);self.assertFalse(bool(cb.prospective(p2).any()))
  for bad in(t._replace(execution=t.execution._replace(party_orders=None)),t._replace(cursor=t.cursor._replace(turn=t.cursor.turn+1))):
   _,q=cb.begin(bad,s,actor,move,self.runtime['direct_hit_context'],self.table,eligible=yes,primary=yes,secondary=yes);self.assertFalse(bool(cb.prospective(q).any()))
  _,called=cb.begin(t,s,actor,move,self.runtime['direct_hit_context'],self.table,eligible=yes,primary=~yes,secondary=yes);self.assertFalse(bool(cb.prospective(called).any()))
  bad_after=after._replace(status=after.status.at[0,actor[0],0].set(3));refused,c=cb.finish(t,p,bad_after,mask);self.assertFalse(bool(cb.captured_matches(refused,c)[0]));self.assertTrue(bool(cb.captured_matches(refused,c)[1:].all()))
  for bad in (p._replace(permutation=p.permutation.at[0,5].set(-1)),p._replace(permutation=p.permutation.at[0,1].set(4))):
   r,c2=cb.finish(t,bad,after,mask);self.assertFalse(bool(cb.captured_matches(r,c2)[0]));self.assertTrue(bool(cb.captured_matches(r,c2)[1:].all()))
  late=t._replace(cursor=t.cursor._replace(turn=t.cursor.turn+1));r,c2=cb.finish(late,p,after,mask);self.assertFalse(bool(cb.captured_matches(r,c2).any()))
  for bad in (done._replace(count=done.count-1),done._replace(words=done.words.at[0,4,6].add(1))):self.assertFalse(bool(cb.owned(bad,3)[0]))
  # Unrevealed cured bench remains UNKNOWN; no private roster alias creation.
  unseen=s._replace(status=s.status.at[rows,actor,3].set(3),ability=s.ability.at[rows,actor,3].set(0));_,q=cb.begin(t,unseen,actor,move,self.runtime['direct_hit_context'],self.table,eligible=yes,primary=yes,secondary=yes);self.assertFalse(bool(cb.prospective(q).any()))

 def test_original_callback_and_both_full_exact_words(self):
  s,t,consumers,actor=self.component();b=8;rows=j.arange(b);yes=j.ones(b,bool);move=j.full(b,int(self.helper.device.heal_bell.move),j.int32)
  t=public.move(t,actor,s.active[rows,actor],s.active[rows,1-actor],move,eligible=yes);t,p=cb.begin(t,s,actor,move,self.runtime['direct_hit_context'],self.table,eligible=yes,primary=yes,secondary=yes)
  after,mask=effects.cure_team(s,actor,self.table.effects,eligible=yes,_return_write_mask=True);t,c=cb.finish(t,p,after,mask);self.assertTrue(bool(cb.captured_matches(t,c).all()))
  frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(consumers[0][0].causal.cursor),life.snapshot(t.cursor),j.zeros(b,bool),j.zeros(b,j.int32));records=[]
  for owner,(stream,pub)in enumerate(consumers):
   r=self.fold(stream,frame,pub);records.append(dict(owner=owner,error=r.projection.error.tolist(),words=r.projection.frames.words.tolist(),valid=r.projection.frames.valid.tolist()));Path('/tmp/pokezero-heal-bell-full-current.json').write_text(json.dumps(records)+'\n');self.assertEqual(r.projection.error.tolist(),[0]*b)
   # The owning accounting closure must grant stage3 only after real FULL.
   ledger,out=ex.consume_frame_device(t.execution.ledger,frame,stream,pub,active=yes,tables=t.execution.tables,adapter_tables=self.runtime['public'].device,catalog=self.catalog)
   self.assertTrue(bool((ledger.stage[:,0]==3).all()))
   for row in range(b):
    side=int(actor[row]);species_name=lambda engine:self.catalog.species.names[int(self.runtime['public'].device.causal.species[engine])];name=species_name(1);bench=species_name(2);ids={i:f'p{i+1}a: {name}'for i in(0,1)}
    # Original mutable-party callback invoked on literal inert Pokemon stubs.
    script=r'''const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),{Pokemon}=require(process.argv[1]+'/dist/sim/pokemon.js'),v=JSON.parse(process.argv[2]),lines=[],b={add:(...x)=>lines.push('|'+x.map(String).join('|')),suppressingAbility:()=>false},mons=v.map(x=>({...x,toString(){return this.ident},hasAbility(n){return n==='soundproof'&&this.sp},removeVolatile(){return false},setStatus(x){this.status=x},battle:b,cureStatus:Pokemon.prototype.cureStatus}));const source=mons[0];source.side={pokemon:mons};d.moves.get('healbell').onHit.call(b,source,source);console.log(JSON.stringify(lines));'''
    ordered=[1,0,2,3,4,5];values=[]
    for physical in ordered:
     active=physical==1;values.append(dict(ident=ids[side]if active else f'p{side+1}: '+bench if physical==0 else f'p{side+1}: Unseen{physical}',hp=int(s.hp[row,side,physical]),status=('', 'par','brn','psn','tox','slp','frz')[int(s.status[row,side,physical])],sp=bool(self.table.effects.soundproof[s.ability[row,side,physical]]),isActive=active))
    lines=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root()),json.dumps(values)],text=True))
    prior=[]
    for who in(0,1):prior.append(f'|switch|p{who+1}a: {bench}|{bench}, L80, M|'+('0 fnt'if int(s.hp[row,who,0])==0 else'100/100 brn'))
    wire,_=public_wire.project_lines(public_wire.WireState(),prior+['|turn|1'],**v4.wire_kwargs(self.catalog))
    current=[f'|switch|{ids[who]}|{name}, L80, M|100/100 par'for who in(0,1)]
    # Row3 explicitly has a healthy own party before its source callback.
    if row==3:current[side]=current[side].replace(' par','');prior[side]=prior[side].replace(' brn','');wire,_=public_wire.project_lines(public_wire.WireState(),prior+['|turn|1'],**v4.wire_kwargs(self.catalog))
    _,packets=public_wire.project_lines(wire,current+[f'|move|{ids[side]}|Heal Bell|{ids[side]}']+lines,**v4.wire_kwargs(self.catalog));want=np.array([e.words()for p in packets for e in p.events],np.int32);actual=np.array(r.projection.frames.words[:,row])[np.array(r.projection.frames.valid[:,row])];Path('/tmp/pokezero-heal-bell-last-oracle.json').write_text(json.dumps(dict(owner=owner,row=row,lines=lines,actual=actual.tolist(),expected=want.tolist()))+'\n');np.testing.assert_array_equal(actual,want)
