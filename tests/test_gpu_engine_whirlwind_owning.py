"""One original quiet drag component, never a game/throughput qualification."""
import ast,builtins,json,subprocess,symtable,unittest
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_whirlwind_callbacks as cb,engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk,engine_request_order as order
from gpu_engine import engine_full_v4_event_device as full,engine_source_frame_witness as witness
from gpu_engine import engine_retained_encoder_driver as driver,phazing,state,public_wire,v4_encoder as v4
import test_gpu_engine_before_move_hidden_power_alias as fixture
from _showdown_root import showdown_root
BASE='e481228626c9cfdfe2a2989871672392aa3e5859'

class WhirlwindOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.BeforeMoveHiddenPowerAliasTests.setUpClass();f=fixture.BeforeMoveHiddenPowerAliasTests
  for n in('metadata','catalog','table','producer','rules','helper','runtime','prepared'):setattr(cls,n,getattr(f,n))
  cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))
  cls.data=cls.helper.device.whirlwind;cls.move=int(cls.data.move)

 def literal(self,consume=False,**kw):
  b=8;r=j.arange(b);side=r%2;z=j.zeros(b,j.int32);yes=j.ones(b,bool);d=self.data
  quiet=int(np.flatnonzero(np.all(np.asarray(d.abilities),axis=0)&(np.arange(d.abilities.shape[1])>0))[0]);species=j.broadcast_to(j.arange(1,7,dtype=j.int32)[None,None,:],(b,2,6));normal=self.metadata['vocabulary']['types'].index('normal')
  s=state.empty_state(b)._replace(species=species,level=j.full((b,2,6),80,j.int32),gender=j.ones((b,2,6),j.int32),done=~yes,hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),ability=j.full((b,2,6),quiet,j.int32),base_ability=j.full((b,2,6),quiet,j.int32),types=j.full((b,2,6,2),normal,j.int32),base_types=j.full((b,2,6,2),normal,j.int32))
  s=s._replace(moves=s.moves.at[r,side,z,0].set(self.move),pp=s.pp.at[r,side,z,0].set(5),boosts=s.boosts.at[r,1-side,z,0].set(2),substitute_hp=s.substitute_hp.at[r,1-side,z].set(25))
  cursor=life.register(r+95001,catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32));initial=life._tape(cursor,self.producer,64)
  for owner in(0,1):initial=life._reveal(initial,s,z+owner,z-1,z,eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=life._turn(initial._replace(cursor=initial.cursor._replace(initialized=yes)),yes);frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),~yes,z);consumers=[]
  for owner in((0,1)if consume else()):
   stream=full.register(cursor.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=z+owner);got=self.fold(stream,frame,pub);self.assertEqual(got.projection.error.tolist(),[0]*b);consumers.append((got.projection.state,got.public))
  t=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;t=ex.begin_on_tape(t,ex.initialize(t.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(t.stream_id,rt),t,rt,active=yes)
  orders=tuple(order.initialize_order(t.stream_id,z+owner,z+6,engine_binding=j.broadcast_to(t.engine_binding,(b,8)))for owner in(0,1));t=t._replace(execution=t.execution._replace(risks=risk.Context(carry,rt),party_orders=orders));t=life.begin(t,s,yes,~yes);move=z+self.move;t=public.move(t,side,z,z,move,eligible=yes)
  args=dict(eligible=yes,hit=yes,immune=~yes,protected=~yes,absorbed=~yes);args.update(kw)
  t,p=cb.begin(t,s,side,move,move,z,r%5,self.runtime['direct_hit_context'],self.table,**args)
  return s,t,p,consumers

 def run_write(self,s,t,p):
  s,changed,ko,t=phazing.drag(s,1-p.side,p.move,p.roll,self.table.phazing,self.table.entry,eligible=p.requested&p.hit,_public_trace=t,_whirlwind_pending=p)
  return s,t,changed,ko

 def test_same_write_mixed_rows_strict_source_lifetime_and_default(self):
  # Keep the shared runtime closed; compile the witness tables only for this raw-tape check, in this test, so a
  # refusing source witness (stale source pins) fails this test alone rather than the whole class.
  s,t,p,_=self.literal();after,done,changed,ko=self.run_write(s,t,p);self.assertEqual(changed.tolist(),[True]*8);self.assertFalse(bool(ko.any()));self.assertEqual(cb.owned(done,p.start).tolist(),[True]*8);self.assertEqual(cb.owned(done,p.start+1).tolist(),[True]*8);self.assertTrue(bool(witness._raw_complete(done,witness.compile_witness(self.prepared).device).all()))
  rec=done.execution.whirlwind[0]
  for name,value in [('caller',p.caller.at[0].add(1)),('roll',p.roll.at[0].set(60)),('hit',p.hit.at[0].set(False)),('source',p.source.at[0,0].add(1)),('stream',p.stream.at[0].add(1)),('generation',p.generation.at[0,0].add(1))]:
   bad=done._replace(execution=done.execution._replace(whirlwind=(rec._replace(pending=p._replace(**{name:value})),)));self.assertEqual(cb.owned(bad,p.start).tolist(),[False]+[True]*7,name)
  for name,value in [('hp',rec.after.hp.at[0,0,0].add(-1)),('pp',rec.after.pp.at[0,0,0,0].add(1)),('status',rec.after.status.at[0,0,0].set(1))]:
   bad=done._replace(execution=done.execution._replace(whirlwind=(rec._replace(final=rec.final._replace(**{name:value})),)));self.assertEqual(cb.owned(bad,p.start+1).tolist(),[False]+[True]*7,name)
  for name in('engine_binding','source_binding','move_binding','producer_binding'):
   with self.assertRaises(ValueError):cb.owned(done._replace(**{name:j.broadcast_to(getattr(done,name),(8,8))}),p.start)
  for bad in(t._replace(execution=t.execution._replace(party_orders=None)),t._replace(refusal=t.refusal|1024)):
   self.assertFalse(bool(cb._start_valid(bad,p)[0].any()))
  # No-source legacy direct compiler has no positive phazing capability.
  from gpu_engine import engine_direct_hit_controls as hit
  self.assertIsNone(hit.compile_direct_hit_controls(engine_catalog=self.producer).device.whirlwind)
  # Actual production operand names/types remain lexical locals, including the
  # B selected slot and scalar-per-row protection/absorption flags.
  text=Path('gpu_engine/transition.py').read_text();st=symtable.symtable(text,'transition','exec');names={x.get_name()for x in st.get_symbols()if x.is_assigned()or x.is_imported()}|set(dir(builtins))|{'__name__','__file__'};todo=[next(x for x in st.get_children()if x.get_name()=='step_with_chance')];bad=set()
  while todo:
   scope=todo.pop();todo.extend(scope.get_children());bad|={x.get_name()for x in scope.get_symbols()if x.is_global()and x.is_referenced()and x.get_name()not in names}
  self.assertEqual(bad,set());call=next(n for n in ast.walk(ast.parse(text))if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='begin'and isinstance(n.func.value,ast.Name)and n.func.value.id=='whirlwind_callbacks');self.assertEqual([ast.unparse(x)for x in call.args[4:7]],['caller_move','chosen_slot','phazing_draws[rows, side]']);self.assertEqual({x.arg:ast.unparse(x.value)for x in call.keywords},dict(eligible='acts',hit='hit',immune='immune',protected='_risk_protected',absorbed='_risk_absorbed'))
  from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/transition.py'],text=True));new=ast.parse(subprocess.check_output(['git','show','448e93f2:gpu_engine/transition.py'],text=True));baseline=dump_default_program(old)
  # Reverse ONLY the additive exact observer block to the original drag call.
  class Strip(ast.NodeTransformer):
   def visit_If(self,node):
    if ast.unparse(node.test)=='_lifecycle_trace and _direct_hit_registered and (_direct_hit_context.data.whirlwind is not None)':return []
    if ast.unparse(node.test)=='_whirlwind_pending is not None':return node.orelse
    return self.generic_visit(node)
   def visit_Assign(self,node):
    return []if any(isinstance(x,ast.Name)and x.id=='_whirlwind_pending'for x in node.targets)else self.generic_visit(node)
  stripped=Strip().visit(new);self.assertEqual(dump_default_program(stripped),baseline);assert_hp_rng_mutations_detected(self,stripped,baseline)
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.

 def test_original_source_drag_writer_and_closed_paths(self):
  script=r"""
const root=process.argv[1],A=require(root+'/dist/sim/battle-actions.js').BattleActions,P=require(root+'/dist/sim/pokemon.js').Pokemon,crypto=require('crypto'),out=[];
for(const variant of['positive','blocked','missing']){
 const events=[],side={active:[],pokemon:[]};
 const battle={gen:3,queue:{cancelAction(p){events.push('cancel:'+p.name)},insertChoice(x){events.push('queue:'+x.choice)}},runEvent(k){events.push(k);return variant!=='blocked'||k!=='DragOut'},singleEvent(k){events.push(k)},initEffectState(x){return x},add(k,p,details){events.push('|'+k+'|'+p+'|'+details.call(p)+'|'+p.hp+'/100')}};
 const old={name:'Old',position:0,hp:100,side,isActive:true,ability:'runaway',baseAbility:'runaway',abilityState:{},itemState:{},baseMoveSlots:[],baseHpType:'Dark',baseHpPower:70,volatiles:{substitute:{}},species:{name:'Old'},setSpecies(){},getAbility(){return{}},getItem(){return{}},battle};old.clearVolatile=P.prototype.clearVolatile;
 const incoming={name:'New',position:1,hp:80,side,isActive:false,moveSlots:[],ability:'runaway',item:'',getFullDetails(){return'New, L80, M'},toString(){return'p1a: New'},battle};side.active=[old];side.pokemon=[old,incoming];battle.getRandomSwitchable=()=>variant==='missing'?null:incoming;
 const actions={battle,switchIn:A.prototype.switchIn};const result=A.prototype.dragIn.call(actions,side,0);out.push({variant,result,events,hp:[old.hp,incoming.hp],active:side.active[0].name,boosts:old.boosts||null,volatiles:old.volatiles});
}
console.log(JSON.stringify(out));"""
  rows=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));positive=rows[0];self.assertTrue(positive['result']);self.assertEqual(positive['hp'],[100,80]);self.assertEqual(positive['active'],'New');self.assertEqual(positive['boosts'],{'atk':0,'def':0,'spa':0,'spd':0,'spe':0,'accuracy':0,'evasion':0});self.assertEqual(positive['volatiles'],{});self.assertIn('|drag|p1a: New|New, L80, M|80/100',positive['events']);self.assertLess(positive['events'].index('|drag|p1a: New|New, L80, M|80/100'),positive['events'].index('queue:runSwitch'))
  for row in rows[1:]:self.assertFalse(row['result']);self.assertEqual(row['active'],'Old');self.assertFalse(any(x.startswith('|drag|')for x in row['events']))

 def test_requested_unsupported_rows_and_source_capability_remain_closed(self):
  from unittest.mock import patch
  s,t,p,_=self.literal();d=self.data
  self.assertFalse(bool(cb._bound(t,d._replace(move=d.move+1)).any()))
  mask=d.abilities.at[0,0].set(~d.abilities[0,0]);self.assertFalse(bool(cb._bound(t,d._replace(abilities=mask)).any()))
  with self.assertRaises(ValueError):cb._bound(t,d._replace(engine_binding=j.broadcast_to(d.engine_binding,(8,8))))
  with patch.object(cb,'SOURCE_HASHES',{**cb.SOURCE_HASHES,'dragIn':'0'*64}),self.assertRaisesRegex(ValueError,'canonical original'):
   cb.compile_data(self.producer,self.metadata,self.rules)
  # Actual no-bench, called and hazard operations still enter Site43 and keep
  # the requested failure, with other rows retaining their real positive claim.
  for kind in('no_bench','called','hazard'):
   before=s
   if kind=='no_bench':before=s._replace(hp=s.hp.at[0,1-int(p.side[0]),1:].set(0))
   if kind=='hazard':before=s._replace(spikes=s.spikes.at[0,1-int(p.side[0])].set(1))
   pending=p._replace(before=before,caller=p.caller.at[0].add(1)if kind=='called'else p.caller)
   after,done,changed,ko=self.run_write(before,t,pending);self.assertEqual(cb.owned(done,p.start).tolist(),[False]+[True]*7,kind)
   l=done.execution.ledger;stages=l.stage[j.arange(8),p.ticket.slot];self.assertEqual(stages.tolist(),[1]+[2]*7,kind);self.assertNotEqual(int(l.refusal[0]),0)
  # Mutating either typed transaction word coherently with ledger copies does
  # not replace the immutable same-call Pending's expected ten-field grammar.
  _,done,_,_=self.run_write(s,t,p)
  for offset,column in((0,5),(0,6),(0,7),(1,4),(1,7)):
   at=int(p.start[0])+offset;words=done.words.at[0,at,column].add(1);l=done.execution.ledger._replace(words=done.execution.ledger.words.at[0,at,column].add(1));bad=done._replace(words=words,execution=done.execution._replace(ledger=l));self.assertEqual(cb.owned(bad,p.start+offset).tolist(),[False]+[True]*7,(offset,column))

 def test_default_phazing_program_exact_and_current_pins(self):
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/phazing.py'],text=True));new=ast.parse(Path('gpu_engine/phazing.py').read_text())
  class Strip(ast.NodeTransformer):
   def visit_If(self,n):
    if ast.unparse(n.test)=='_public_trace is not None':return n.orelse
    return self.generic_visit(n)
   def visit_FunctionDef(self,n):
    if n.name=='drag':n.args.kwonlyargs=n.args.kwonlyargs[:1];n.args.kw_defaults=n.args.kw_defaults[:1]
    return self.generic_visit(n)
  stripped=Strip().visit(new);self.assertEqual(ast.dump(stripped,include_attributes=False),ast.dump(old,include_attributes=False))
  poisoned=ast.parse(ast.unparse(stripped).replace('eligible & table.move[move]','eligible & False & table.move[move]'));self.assertNotEqual(ast.dump(poisoned,include_attributes=False),ast.dump(old,include_attributes=False))
  witness._source()
  import hashlib
  for file,key,count in(('chance_consumers.py','SOURCE_PINS',14),('ordinary_law.py','MAPPING_PINS',6),('conditional_law.py','MAPPING_PINS',4)):
   tree=ast.parse(Path('gpu_engine',file).read_text());mapping=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id==key for x in n.targets)));self.assertEqual(len(mapping),count)
   for name,value in mapping.items():self.assertEqual(hashlib.sha256(Path('gpu_engine',name).read_bytes()).hexdigest(),value)
  self.assertEqual(len(witness.SOURCE_PINS),89)

 def test_original_both_owner_drag_words(self):
  s,t,p,consumers=self.literal(True);before_registry=life.snapshot(t.cursor);after,t,changed,ko=self.run_write(s,t,p);yes=j.ones(8,bool);no=~yes;z=j.zeros(8,j.int32);t=life.action_end(t,yes);t=life.finish(t,after,live=yes,replacement=no,pending_before=no,residual_ready=yes,defer_residual=no,turn_context=yes);frame=life.LifecycleResult(after,t.cursor,t,before_registry,life.snapshot(t.cursor),no,z);packets=[]
  for owner,(stream,pub)in enumerate(consumers):
   got=self.fold(stream,frame,pub)
   for row in range(8):
    actor=int(p.side[row]);foe=1-actor;incoming=int(after.active[row,foe])
    # Use the same actual source species names and new incoming physical fact.
    names={side:self.catalog.species.names[self.runtime['public'].device.causal.species[int(s.species[row,side,0])]]for side in(0,1)};incoming_name=self.catalog.species.names[int(self.runtime['public'].device.causal.species[int(after.species[row,foe,incoming])])];idents={side:f'p{side+1}a: '+names[side]for side in(0,1)}
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{idents[side]}|{names[side]}, L80, M|100/100'for side in(0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));_,expected=public_wire.project_lines(wire,['|','|t:|1',f'|move|{idents[actor]}|Whirlwind|{idents[foe]}',f'|drag|p{foe+1}a: {incoming_name}|{incoming_name}, L80, M|100/100','|','|upkeep','|turn|2'],**v4.wire_kwargs(self.catalog));wanted=np.asarray([event.words()for packet in expected for event in packet.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];packets.append(dict(owner=owner,row=row,error=int(got.projection.error[row]),admitted=bool(got.projection.component_admitted[row]),actual=actual.tolist(),expected=wanted.tolist()))
  Path('/tmp/pokezero-whirlwind-original-both-full-r1.json').write_text(json.dumps(packets)+'\n')
  for packet in packets:self.assertEqual(packet['error'],0);self.assertTrue(packet['admitted']);np.testing.assert_array_equal(packet['actual'],packet['expected'])

if __name__=='__main__':unittest.main()
