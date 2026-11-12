"""Prospective Color Change components; no scientific worlds/models/roots."""
from copy import deepcopy
import ast,json,os,subprocess,unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import gc
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import secondary,entry,phazing,state as states,transition as tr,public_belief,v4_encoder as v4,public_wire,source_provenance
from gpu_engine import engine_color_change_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixtures
import gpu_engine_eager_witness as eager

class ColorChangeOwningTests(unittest.TestCase):
 def setUp(self):
  self.release_compilers()
 @staticmethod
 def release_compilers(*completed):
  # Keep the exact arrays, but release completed CPU executable heaps before
  # compiling the next genuine phase. This is a correctness fixture only.
  jax.block_until_ready(completed);jax.clear_caches();gc.collect()
 def completed_phase(self,name,_milliseconds):
  # Source-witness mode completes both owner folds in public_pair.
  if name in ('engine','requests','public_pair','public_p2','admission_p2'):
   self.release_compilers()
   print(type(self).__name__+' completed CPU phase '+name,flush=True)
 def construct(self,*args,**kwargs):
  original=driver._consume
  def initial_consume(runtime,state,frame,*a,**kw):
   self.release_compilers(state,frame)
   print(type(self).__name__+' constructor inputs ready',flush=True)
   return original(runtime,state,frame,*a,**dict(kw,timing_callback=self.completed_phase))
  with patch.object(driver,'_consume',side_effect=initial_consume):
   return driver.construct(*args,**kwargs)
 @classmethod
 def setUpClass(cls):
  fixtures.NoDamageOwningTests.setUpClass();cls.f=fixtures.NoDamageOwningTests;cls.metadata=deepcopy(cls.f.metadata)
  native=load_showdown_dex_cached(showdown_root());moves=dict(cls.f.dex.moves);species=dict(cls.f.dex.species);cls.names=('surf','tackle','protect','roar')
  for name in cls.names:
   info=native.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   i=cls.metadata['vocabulary']['moves'].index(name)
   cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=True if info.accuracy is True else int(info.accuracy),priority=info.priority,flags={'protect':1}if name in ('surf','tackle')else{})
  for name in ('kecleon','blastoise'):
   if name not in cls.metadata['vocabulary']['species']:cls.metadata['vocabulary']['species'].append(name)
   species[name]=native.species_info(name)
  if 'colorchange'not in cls.metadata['vocabulary']['abilities']:cls.metadata['vocabulary']['abilities'].append('colorchange')
  cls.dex=ShowdownDex(moves,species,native.type_chart)
  variants={name:Gen3RandbatSpeciesUniverse(name,80,tuple(Gen3RandbatVariant('literal-color-'+name+str(i),'authored',name,'authored',80,cls.names,ability,'Leftovers')for i,ability in enumerate(('Pressure','Color Change'))))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-color-change-source'),variants)
  labels=set(cls.f.catalog.vocabulary.tokens)|{'ability:colorchange','belief:possible_ability:colorchange','species:kecleon','species:blastoise'}
  for name in cls.names:labels|={'move:'+name,'belief:possible_move:'+name}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=cls.f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer)
  cls.facts=own.compile_own_facts(cls.req,cls.catalog);cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(cls.req,cls.facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.water=cls.metadata['vocabulary']['types'].index('water');cls.normal=cls.metadata['vocabulary']['types'].index('normal');cls.color=cls.metadata['vocabulary']['abilities'].index('colorchange');cls.pressure=cls.metadata['vocabulary']['abilities'].index('pressure');cls.surf=cls.metadata['vocabulary']['moves'].index('surf')
 def test_effective_handler_literal_guards_type_write_and_all_departure_paths(self):
  script=r"""const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),a=d.abilities.get('colorchange'),out=[];
for(const [damage,hp,active,same,category,type]of[[10,100,true,false,'Special','Water'],[0,100,true,false,'Special','Water'],[10,0,true,false,'Special','Water'],[10,100,false,false,'Special','Water'],[10,100,true,true,'Special','Water'],[10,100,true,false,'Status','Water'],[10,100,true,false,'Special','???']]){
const lines=[],p={hp,isActive:active,hasType:()=>same,setType(t){this.type=t;return true},toString:()=> 'p2a: Kecleon'};a.onDamagingHit.call({add:(...x)=>lines.push(x.map(String).join('|'))},damage,p,{}, {category,type});out.push({type:p.type||null,lines});}process.stdout.write(JSON.stringify({after:a.onAfterMoveSecondary===undefined,out}));"""
  original=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertTrue(original['after']);self.assertEqual(original['out'][0],dict(type='Water',lines=['-start|p2a: Kecleon|typechange|Water|[from] ability: Color Change']));self.assertTrue(all(not x['lines']for x in original['out'][1:]))
  b=8;s=states.empty_state(b);base=j.broadcast_to(j.array([self.normal,0]),s.types.shape);side=j.arange(b)%2;rows=j.arange(b);target=1-side
  s=s._replace(types=base,base_types=base,hp=j.ones_like(s.hp)*100,ability=j.ones_like(s.ability)*self.color,base_ability=j.ones_like(s.ability)*self.color)
  s=s._replace(hp=s.hp.at[2,target[2],0].set(0),types=s.types.at[3,target[3],0].set(j.array([self.water,0])),ability=s.ability.at[4,target[4],0].set(self.pressure))
  eligible=j.ones(b,bool).at[1].set(False);typ=j.full(b,self.water,j.int32).at[5].set(0)
  result=jax.jit(lambda x:secondary.apply_color_change(x,side,typ,self.table.secondary,eligible=eligible))(s)
  expected=[self.water,self.normal,self.normal,self.water,self.normal,self.normal,self.water,self.water]
  self.assertEqual(result.types[rows,target,0,0].tolist(),expected)
  for key in s._fields:
   if key!='types':np.testing.assert_array_equal(getattr(result,key),getattr(s,key),err_msg=key)
  # All current switch kinds converge on this exact shared cleanup; inactive
  # members keep their changed type. Trace-current ability does not skip reset.
  departed=entry.depart(result,target,self.table.entry,eligible=j.arange(b)%2==0)
  want=np.asarray(result.types).copy()
  for row in range(b):
   if row%2==0:want[row,int(target[row]),0]=[self.normal,0]
  np.testing.assert_array_equal(departed.types,want);np.testing.assert_array_equal(departed.base_types,base)
  drag_table=self.table.phazing;roar=j.full(b,self.metadata['vocabulary']['moves'].index('roar'),j.int32)
  dragged,changed,_=phazing.drag(result,target,roar,j.zeros(b,j.int32),drag_table,self.table.entry,eligible=j.ones(b,bool))
  self.assertTrue(bool(changed[0]));self.assertEqual(dragged.types[0,int(target[0]),0].tolist(),[self.normal,0])
  with self.assertRaises(ValueError):direct.check_compiled(direct.Compiled(self.helper.device))
  with self.assertRaises(ValueError):jax.jit(lambda:direct.check_compiled(self.helper).binding)()
 def test_exact_default_program_no_direct_trace_and_missing_hypothetical_base_refusal(self):
  from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
  old=ast.parse(subprocess.check_output(['git','show','c41d55db7cfa79b9043ad179a2231faee81e66fa:gpu_engine/transition.py'],text=True))
  class IntendedTypes(ast.NodeTransformer):
   def visit_Assign(self,node):
    if isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Attribute)and node.value.func.attr=='apply_color_change':return None
    return self.generic_visit(node)
   def visit_If(self,node):
    node=self.generic_visit(node)
    if not node.body:
     if node.orelse:return node.orelse
     return None
    return node
   def visit_Call(self,node):
    node=self.generic_visit(node)
    if isinstance(node.func,ast.Attribute)and node.func.attr=='_replace':
     for key in node.keywords:
      if key.arg=='base_types':
       if ast.dump(key.value)!=ast.dump(ast.parse('values["types"]',mode='eval').body):raise AssertionError('constructor type storage expression changed')
     node.keywords=[x for x in node.keywords if x.arg!='base_types']
    return node
  normalized=IntendedTypes().visit(ast.parse(subprocess.check_output(['git','show','1da1c80d:gpu_engine/transition.py'],text=True)));baseline=dump_default_program(old)
  self.assertEqual(dump_default_program(normalized),baseline);assert_hp_rng_mutations_detected(self,normalized,baseline)
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
  s=states.empty_state(1);base=j.broadcast_to(j.array([self.normal,0]),s.types.shape)
  s=s._replace(types=base,base_types=base,species=j.ones_like(s.species),level=j.ones_like(s.level)*80,hp=j.ones_like(s.hp)*100,max_hp=j.ones_like(s.hp)*100,stats=j.ones_like(s.stats)*100,ability=j.ones_like(s.ability)*self.pressure,base_ability=j.ones_like(s.ability)*self.pressure,moves=j.ones_like(s.moves)*self.surf,pp=j.ones_like(s.pp)*24,entry_pending=j.zeros_like(s.entry_pending))
  cursor=life.register(j.array([59001],j.int32),catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));tape=life._tape(cursor,self.producer,64)
  chance=tr.ChanceInputs(j.zeros((1,2,12),j.int32),*[j.zeros((1,2),j.int32)for _ in range(4)],j.zeros(1,j.int32))
  shaped=jax.eval_shape(lambda x:tr.step_with_chance(x,j.zeros((1,2),j.int32),self.table,chance,_public_trace=tape,_lifecycle_trace=True),s)
  self.assertEqual(shaped.step.state.base_types.shape,(1,2,6,2))
  from test_gpu_engine_v5_public_world import TypedMappingTests
  from gpu_engine import v5_hypothetical_state as mapping
  spec,metadata=TypedMappingTests().fixture();mon=spec.side_one.pokemon[0]
  missing=replace(mon,types=('ice',),base_types=None);side=replace(spec.side_one,pokemon=(missing,));changed=replace(spec,side_one=side)
  result=mapping.map_spec(changed,metadata=metadata,seed=1,turn=1,active_turns=(0,0),happiness=((7,),(255,)),dex=load_showdown_dex_cached(showdown_root()))
  self.assertIsNone(result.state);self.assertIn('changed-types-base-unavailable',result.refusals)
  original=mapping.map_spec(spec,metadata=metadata,seed=1,turn=1,active_turns=(0,0),happiness=((7,),(255,)))
  self.assertFalse(original.refusals);self.assertEqual(original.state.base_types[0,0,0].tolist(),[1,0])
  normal=replace(mon,base_types=None);normal_spec=replace(spec,side_one=replace(spec.side_one,pokemon=(normal,)))
  mapped=mapping.map_spec(normal_spec,metadata=metadata,seed=1,turn=1,active_turns=(0,0),happiness=((7,),(255,)),dex=load_showdown_dex_cached(showdown_root()))
  self.assertFalse(mapped.refusals);self.assertEqual(mapped.state.base_types[0,0,0].tolist(),[1,0])
  forged=replace(missing,base_types=('ice',));forged_spec=replace(spec,side_one=replace(spec.side_one,pokemon=(forged,)))
  rejected=mapping.map_spec(forged_spec,metadata=metadata,seed=1,turn=1,active_turns=(0,0),happiness=((7,),(255,)),dex=load_showdown_dex_cached(showdown_root()))
  self.assertIsNone(rejected.state);self.assertIn('base-types-differ-original-species',rejected.refusals)

 def component(self):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);side=rows%2;active=j.stack((z+3,z+4),axis=1)
  s=states.empty_state(b);base=j.broadcast_to(j.array([self.normal,0]),s.types.shape)
  s=s._replace(types=base,base_types=base,active=active,species=j.broadcast_to(j.array([1,2])[None,:,None],s.species.shape),hp=j.ones_like(s.hp)*100,max_hp=j.ones_like(s.hp)*100,level=j.ones_like(s.level)*80,gender=j.ones_like(s.gender),ability=j.ones_like(s.ability)*self.color,base_ability=j.ones_like(s.ability)*self.color)
  kw=dict(catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));cursor=life.register(rows+57001,**kw);initial=life._tape(cursor,self.producer,64)
  for owner in (0,1):initial=life._reveal(initial,s,z+owner,z-1,active[:,owner],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes)
  frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,z);consumers=[]
  fold=jax.jit(lambda stream,frame,pub:full.project_and_fold(stream,frame,self.runtime['public'].device,pub,self.catalog))
  for owner in (0,1):
   stream=full.register(cursor.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=z+owner);r=fold(stream,frame,pub);self.assertEqual(r.projection.error.tolist(),[0]*b);consumers.append((r.projection.state,r.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rt),tape,rt,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rt)))
  physical=active[rows,side];target=active[rows,1-side];move=z+self.surf;tape=public.move(tape,side,physical,target,move,eligible=yes)
  tape,ticket=ex.open_attempt(tape,ex.Site.DIRECT_HIT,1-side,active=yes)
  w=direct.HitWrite(z+95,~yes,yes,~yes,z+1,z+1,~yes,~yes,~yes,z+100,z+90,z+100)
  ctx=self.runtime['direct_hit_context'];tape=direct.record_registered_direct_hit_controls_device(tape,side,physical,target,move,w,ctx,eligible=yes);tape=public.condition(tape,1-side,target,z+90,z+100,z,eligible=yes)
  tape=ex.capture_direct_hit_write(tape,ticket,side,physical,target,move,w,requested=yes,data=ctx.data)
  s=s._replace(hp=s.hp.at[rows,1-side,target].set(90))
  tape=tape._replace(refusal=j.array([1]*7+[257],j.int32));requested=yes.at[6].set(False)
  tape,pending=cb.begin(tape,s,side,move,z+self.water,ctx,self.table.secondary,eligible=requested)
  new=s.types.at[rows,1-side,target].set(j.broadcast_to(j.array([self.water,0]),(b,2)));after=s._replace(types=new);after=after._replace(types=after.types.at[5,1-side[5],target[5]].set(j.array([self.normal,0])))
  tape=cb.finish(tape,pending,after)
  return s,after,tape,pending,consumers,fold
 def test_mixed_literal_receipt_namespaces_sticky_and_original_full_words(self):
  before,after,tape,p,consumers,fold=self.component();self.assertEqual(tape.count.tolist(),[3,3,3,3,3,2,2,3]);self.assertEqual(tape.refusal.tolist(),[1,1,1,1,1,769,1,257]);self.assertEqual(cb.owned(tape,2).tolist(),[True]*5+[False,False,True])
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=tape._replace(**{name:getattr(tape,name).at[0].add(j.uint32(1))});self.assertFalse(bool(cb.owned(bad,2).any()))
   with self.assertRaises(ValueError):cb.owned(tape._replace(**{name:j.broadcast_to(getattr(tape,name),(8,8))}),2)
  frame=life.LifecycleResult(after,tape.cursor,tape,life.snapshot(tape.cursor),life.snapshot(tape.cursor),j.zeros(8,bool),j.zeros(8,j.int32))
  for owner,(stream,pub)in enumerate(consumers):
   result=fold(stream,frame,pub);self.assertEqual(result.projection.error[:5].tolist(),[0]*5)
   names={s:self.metadata['vocabulary']['species'][s+1]for s in (0,1)};ids={s:f'p{s+1}a: '+names[s]for s in (0,1)}
   wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ids[s]}|{names[s]}, L80, M|100/100'for s in (0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog))
   for row in range(5):
    actor=row%2;target=1-actor
    _,packets=public_wire.project_lines(wire,[f'|move|{ids[actor]}|Surf|{ids[target]}',f'|-damage|{ids[target]}|90/100',f'|-start|{ids[target]}|typechange|Water|[from] ability: Color Change'],**v4.wire_kwargs(self.catalog))
    expected=np.asarray([e.words()for packet in packets for e in packet.events],np.int32);actual=np.asarray(result.projection.frames.words[:,row])[np.asarray(result.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,expected)
 def literal_pool_arrays(self):
  arrays=self.f.f.base().arrays('pressure','pressure');arrays['moves'][:]=np.asarray([self.metadata['vocabulary']['moves'].index(n)for n in self.names]);arrays['pp'][:]=[24,56,16,32];arrays['item'][:]=self.metadata['vocabulary']['items'].index('leftovers')
  # Two literal mirrored synthetic rows. The Color Change holder acts first,
  # so the intended type change cannot alter this frame's earlier damage.
  arrays['ability'][:]=self.pressure
  for team in (0,1):arrays['species'][team,0]=self.metadata['vocabulary']['species'].index('kecleon'if team==0 else'blastoise')
  arrays['ability'][0,0]=self.color
  arrays['species'][:,1]=self.metadata['vocabulary']['species'].index('vaporeon')
  dex=load_showdown_dex_cached(showdown_root())
  for team in (0,1):
   self.assertEqual(len(set(arrays['species'][team].tolist())),6)
   for slot in range(6):
    info=dex.species_info(self.metadata['vocabulary']['species'][int(arrays['species'][team,slot])]);types=[self.metadata['vocabulary']['types'].index(t.lower())for t in info.types];arrays['types'][team,slot]=types+[0]*(2-len(types))
  arrays['stats'][:,:,0]=350;arrays['stats'][:,:,1:5]=100;arrays['stats'][0,:,5]=190;arrays['stats'][1,:,5]=110
  return arrays
 def test_literal_pool_host_preflight(self):
  arrays=self.literal_pool_arrays()
  self.assertEqual(arrays['species'].shape,(2,6));self.assertEqual(arrays['types'].shape,(2,6,2))
 def test_actual_original_frame_both_views_public_types_requests_and_physics(self):
  self.enterContext(eager.eager_source_witness(self.prepared))
  arrays=self.literal_pool_arrays();captured={'frames':[]};capture_path=os.environ.get('POKEZERO_COLOR_CHANGE_CAPTURE')
  def snapshot(result,frame=None,physical=None):
   core=result.state.core;engine=core.engine;report=core.frame_input_report
   record=dict(state={key:np.asarray(getattr(engine,key)).tolist()for key in ('species','types','base_types','active','hp','max_hp','done','winner','turn','keys')},core_admitted=core.admitted.tolist(),requests=[x.reason.tolist()for x in core.requests],owners=[dict(reason=o.reason.tolist(),own=o.own.reason.tolist(),admitted=o.encoded.observation.admitted.tolist(),full=o.full_line_consumers_complete.tolist(),encoder=o.full_encoder_admitted.tolist(),basic=o.basic_error.tolist(),retained=o.retained_error.tolist())for o in result.owners])
   if report is not None:
    record.update(witness=report.refusal.tolist(),expected=report.expected_cutpoints.tolist(),observed=report.observed_cutpoints.tolist())
    words=np.asarray(report.owner_full_words);valid=np.asarray(report.owner_full_valid)
    record['owner_full_words']=[[words[row,owner].reshape((-1,words.shape[-1]))[valid[row,owner].reshape(-1)].tolist()for owner in (0,1)]for row in range(2)]
   if core.execution is not None:record['ledger']={key:np.asarray(getattr(core.execution,key)).tolist()for key in ('refusal','site','stage','start','end','count')}
   if core.risks is not None:record['risk']=core.risks.risk.tolist()
   if frame is not None:record.update(raw=[frame.callbacks.words[row,:int(frame.callbacks.count[row])].tolist()for row in range(2)],source=frame.callbacks.refusal.tolist(),stamps=dict(turn=frame.callbacks.event_turn.tolist(),phase=frame.callbacks.event_phase.tolist(),generation=frame.callbacks.event_generation.tolist()))
   if physical is not None:record['physical_equal']={key:bool(np.array_equal(getattr(engine,key),getattr(physical,key)))for key in physical._fields}
   return record
  def save():
   if capture_path:Path(capture_path).write_text(json.dumps(captured,sort_keys=True,allow_nan=False)+'\n')
  with TemporaryDirectory()as tmp:
   path=Path(tmp)/'literal.npz';np.savez_compressed(path,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8)));pool=source_provenance.load_constructor_pool(path)
   session=self.construct(pool,np.array([[0,1],[1,0]],np.int64),prepared=self.prepared,stream_ids=j.array([58001,58002],j.int32),seed=31)
  captured['initial']=snapshot(session.result);save()
  self.assertEqual(session.state.core.admitted.tolist(),[True,True])
  for owner in session.result.owners:self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True],str(owner.reason.tolist()))
  before=session.state;actions=j.array([[1,0],[0,1]],j.int32);draws=j.zeros((2,2,12),j.int32).at[:,:,4].set(1);chance=tr.ChanceInputs(draws,*[j.zeros((2,2),j.int32)for _ in range(4)],j.zeros(2,j.int32));frames=[];original=self.runtime['frame']
  def capture(*args,**kwargs):frame=original(*args,**kwargs);frames.append(frame);return frame
  self.runtime['frame']=capture
  try:result=driver.advance(session,actions,chance_inputs=chance,timing_callback=self.completed_phase)
  finally:self.runtime['frame']=original
  frame=frames[-1];plain=jax.jit(lambda s,a:tr.step_with_chance(s,a,self.table,chance).state)(before.core.engine,actions)
  captured['frames'].append(snapshot(result,frame,plain));save()
  for key in plain._fields:np.testing.assert_array_equal(getattr(frame.state,key),getattr(plain,key),err_msg=key)
  diagnostic=dict(raw=frame.callbacks.words.tolist(),source=frame.callbacks.refusal.tolist(),risk=result.state.core.risks.risk.tolist(),ledger=result.state.core.execution.refusal.tolist(),witness=result.state.core.frame_input_report.refusal.tolist(),expected=result.state.core.frame_input_report.expected_cutpoints.tolist(),actual=result.state.core.frame_input_report.observed_cutpoints.tolist(),requests=[x.reason.tolist()for x in result.state.core.requests],owners=[o.reason.tolist()for o in result.owners])
  self.assertEqual(result.state.core.admitted.tolist(),[True,True],json.dumps(diagnostic))
  for row,target in ((0,0),(1,1)):
   self.assertEqual(frame.state.types[row,target,0].tolist(),[self.water,0]);self.assertEqual(frame.state.base_types[row,target,0].tolist(),[self.normal,0]);raw=np.asarray(frame.callbacks.words[row,:int(frame.callbacks.count[row])]);self.assertEqual(int(np.sum(raw[:,0]==cb.COLOR_CHANGE)),1)
  for owner in result.owners:self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True],json.dumps(diagnostic));self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True]);self.assertFalse(bool(owner.B4_qualified.any()))
  used=result.state.core.execution.site==int(ex.Site.COLOR_CHANGE);self.assertTrue(bool(j.all(j.where(used,result.state.core.execution.stage==3,True))))
  # Prior public type evidence, not the just-finished raw frame, retains the
  # live type for the next owning request. A corrupted encounter cannot do so.
  empty=life._tape(result.state.core.cursor,self.producer,64)
  for owner,observer in enumerate(result.state.observers):
   expected=cb.expected_owned_types(result.state.core.engine,result.state.core.engine,observer.retained,owner=owner,tape=empty,data=self.helper.device.color_change,categories=self.runtime['public'].device.causal.color_categories,type_tables=self.catalog.types.device,healthy=observer.retained_healthy)
   np.testing.assert_array_equal(expected,result.state.core.engine.types[:,owner])
  for frame_index in range(2):
   before=result.state.core.engine;switches=j.full((2,2),4,j.int32)
   result=driver.advance(session,switches,chance_inputs=chance,timing_callback=self.completed_phase)
   plain=jax.jit(lambda s:tr.step_with_chance(s,switches,self.table,chance).state)(before)
   captured['frames'].append(snapshot(result,physical=plain));save()
   for key in plain._fields:np.testing.assert_array_equal(getattr(result.state.core.engine,key),getattr(plain,key),err_msg=key)
   self.assertEqual(result.state.core.admitted.tolist(),[True,True],str([o.reason.tolist()for o in result.owners]))
   self.assertEqual(result.state.core.engine.types[0,0,0].tolist(),[self.normal,0])
   self.assertEqual(result.state.core.engine.types[1,1,0].tolist(),[self.normal,0])
   for owner in result.owners:self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True])


def tearDownModule():
    eager.clear()   # drop the consume executables this module compiled


if __name__=='__main__':unittest.main()
