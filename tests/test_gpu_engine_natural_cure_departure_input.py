"""Actual typed departure helpers, source writer and consumer; no engine turn."""
from _component_evidence import capture_path
import json,subprocess,symtable,unittest
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_natural_cure_callbacks as nc,engine_baton_pass_callbacks as bp,engine_direct_hit_controls as direct
from gpu_engine import engine_lifecycle_callbacks as life,engine_execution_completeness as ex,engine_kernel_frame_risk as risk,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own,engine_leftovers_callbacks as leftovers
from gpu_engine import engine_full_v4_event_device as full,engine_source_frame_witness as witness,public_belief,v4_encoder as v4,state,entry,baton,transition,public_wire,public_inputs as inputs
from _showdown_root import showdown_root
import test_gpu_engine_hidden_power_substitute_input as fixture
BASE='e481228626c9cfdfe2a2989871672392aa3e5859'
class NaturalCureTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.HiddenPowerSubstituteInputTests.setUpClass();f=fixture.HiddenPowerSubstituteInputTests;cls.f=f;cls.metadata=deepcopy(f.metadata);dex=load_showdown_dex_cached(showdown_root());moves=dict(f.dex.moves);cls.names=('batonpass','surf','tailglow','agility')
  for name in cls.names:
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   i=cls.metadata['vocabulary']['moves'].index(name);row=json.loads(subprocess.check_output(['node','-e',"const m=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3').moves.get(process.argv[2]);console.log(JSON.stringify({accuracy:m.accuracy,flags:m.flags}));",str(showdown_root()),name],text=True));cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category=info.category,pp=info.pp,accuracy=row['accuracy'],priority=info.priority,flags=row['flags'])
  allnames=json.loads(subprocess.check_output(['node','-e',"const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),ids=a=>['_none',...Array.from(new Set(a.filter(x=>x.exists&&x.gen<=3).map(x=>x.id))).sort()];console.log(JSON.stringify({abilities:ids(d.abilities.all()),items:ids(d.items.all())}));",str(showdown_root())],text=True));cls.metadata['vocabulary'].update(allnames);cls.ability=allnames['abilities'].index('naturalcure');cls.quiet=allnames['abilities'].index('runaway');cls.move=cls.metadata['vocabulary']['moves'].index('batonpass');cls.dex=ShowdownDex(moves,f.dex.species,f.dex.type_chart)
  variants={n:Gen3RandbatSpeciesUniverse(n,80,tuple(Gen3RandbatVariant('nc-'+n+str(i),'unit',n,'unit',80,cls.names,a,'Leftovers')for i,a in enumerate(('Natural Cure','Run Away','Rough Skin'))))for n in cls.metadata['vocabulary']['species'][1:]};src=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-natural-cure-departure'),variants);labels=set(f.catalog.vocabulary.tokens)|{'ability:naturalcure','belief:possible_ability:naturalcure','ability:runaway','belief:possible_ability:runaway','ability:roughskin','belief:possible_ability:roughskin'}
  for name in cls.names:labels|={'move:'+name,'belief:possible_move:'+name}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(src),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=f.catalog.masks.__dict__);cls.table=transition.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False));cls.rules=request.load_source_rules(showdown_root(),cls.metadata);cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(cls.req,cls.catalog);cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(cls.req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)));cls.runtime=driver._checked(cls.prepared);cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32))
 def component(self,b=8,statuses=None):
  r=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);side=r%2;old=z+3;new=z+4;active=j.zeros((b,2),j.int32).at[r,side].set(old)
  status=j.array([1,2,3,4,5,6,0,4][:b],j.int32)if statuses is None else statuses
  s=state.empty_state(b)._replace(done=~yes,active=active,hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.ones((b,2,6),j.int32),level=j.full((b,2,6),80,j.int32),gender=j.ones((b,2,6),j.int32),ability=j.full((b,2,6),self.quiet,j.int32),base_ability=j.full((b,2,6),self.quiet,j.int32));s=s._replace(ability=s.ability.at[r,side,old].set(self.ability),base_ability=s.base_ability.at[r,side,old].set(self.ability),status=s.status.at[r,side,old].set(status),status_counter=s.status_counter.at[r,side,old].set(j.where(status==4,2,j.where(status==5,3,0))),sleep_skipped=s.sleep_skipped.at[r,side,old].set(j.where(status==5,2,0)),rest_origin=s.rest_origin.at[r,side,old].set(status==5))
  cursor=life.register(r+680001,**self.kw);mapping=j.full((b,2,6),-1,j.int32).at[:,:,0].set(0).at[r,side,old].set(0);cursor=cursor._replace(turn=z+1,initialized=yes,physical_active=active,physical_to_public=mapping,public_slot=j.zeros((b,2),j.int32),generation=j.ones((b,2),j.int32),encounter=j.ones((b,2),j.int32),established=j.ones((b,2),bool),next_slot=j.ones((b,2),j.int32))
  t=life._tape(cursor,self.producer,64);a=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;t=ex.begin_on_tape(t,ex.initialize(t.stream_id,compiled=a),a.device,active=yes);t=t._replace(execution=t.execution._replace(risks=risk.Context(risk.begin_frame_device(risk.initialize_device(t.stream_id,rt),t,rt,active=yes),rt)));return t,s,side,old,new,yes
 def test_actual_departure_all_major_statuses_source_and_current_joins(self):
  t,s,side,old,new,yes=self.component();t,p=nc.begin(t,s,side,new,self.table,self.runtime['direct_hit_context'],eligible=yes,baton=~yes);after=entry.depart(s,side,self.table.entry,eligible=yes);out=nc.finish(t,p,after)
  self.assertEqual(out.count.tolist(),[1]*6+[0,1]);self.assertEqual(out.execution.ledger.site[:,0].tolist(),[45]*6+[0,45]);self.assertEqual(after.status[j.arange(8),side,old].tolist(),[0]*8);self.assertEqual(after.sleep_skipped[j.arange(8),side,old].tolist(),[0]*8)
  for field in ('hp','item','stats','moves','pp','keys'):
   changed=after._replace(**{field:getattr(after,field)+1});bad=nc.finish(t,p,changed);self.assertEqual(bad.count.tolist(),[0]*8,field);self.assertTrue(bool(j.all((bad.refusal[:6]&128)!=0)))
  # Pending.valid never replaces replay of current capability and source facts.
  d=p.context.data;badctx=p.context._replace(data=d._replace(baton_pass=d.baton_pass._replace(natural_cure=j.zeros_like(d.baton_pass.natural_cure))));bad=nc.finish(t,p._replace(context=badctx),after);self.assertEqual(bad.count.tolist(),[0]*8)
  bad=nc.finish(t,p._replace(before=None),after);self.assertEqual(bad.count.tolist(),[0]*8)
  # Original source onSwitchOut+clearStatus supplies each exact public line.
  script=r'''const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),P=require(root+'/dist/sim/pokemon.js').Pokemon,out=[];for(const status of ['par','brn','psn','tox','slp','frz']){const lines=[],target={hp:100,status,clearStatus:P.prototype.clearStatus,removeVolatile:()=>false,setStatus(s){this.status=s},toString:()=> 'p1a: Actor'};d.abilities.get('naturalcure').onSwitchOut.call({add:(...x)=>lines.push('|'+x.map(String).join('|'))},target);out.push({status:target.status,lines});}console.log(JSON.stringify(out));'''
  got=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(got,[dict(status='',lines=[f'|-curestatus|p1a: Actor|{c}|[from] ability: Natural Cure'])for c in ('par','brn','psn','tox','slp','frz')])
 def test_default_physics_source_and_wiring(self):
  witness._source();self.assertEqual((self.ability,len(self.metadata['vocabulary']['abilities'])),(38,81));self.assertEqual(int(ex.Site.NATURAL_CURE),45)
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
  # entry.py, baton.py, state.py and items.py are byte-certified against the reviewed SOURCE_PINS by transition_observer_wiring.
  for name in ('mechanics.py',):self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+name]))
  import builtins
  module=symtable.symtable(Path('gpu_engine/transition.py').read_text(),'transition.py','exec');step=next(c for c in module.get_children()if c.get_name()=='step_with_chance');self.assertEqual([s.get_name()for s in step.get_symbols()if s.is_referenced()and s.is_global()and s.get_name()not in set(module.get_identifiers())|set(dir(builtins))],[])
 def test_ordinary_late_owned_all_statuses_and_receipt_poison(self):
  t,s,side,old,new,yes=self.component();t,p=nc.begin(t,s,side,new,self.table,self.runtime['direct_hit_context'],eligible=yes,baton=~yes);after=entry.depart(s,side,self.table.entry,eligible=yes);t=nc.finish(t,p,after);switched=after._replace(active=after.active.at[j.arange(8),side].set(new));t=life.installed(t,switched,side,old,new,eligible=yes,replacement=~yes,baton=~yes);mask=[True]*6+[False,True]
  self.assertEqual(nc.owned(t,0).tolist(),mask);self.assertTrue(bool(witness._raw_complete(t,self.runtime['frame_witness_compiled'].device).all()))
  for col in range(10):
   badraw=t.words.at[0,0,col].add(1);ledger=t.execution.ledger._replace(words=t.execution.ledger.words.at[0,0,col].add(1));bad=t._replace(words=badraw,execution=t.execution._replace(ledger=ledger));self.assertEqual(nc.owned(bad,0).tolist(),[False]+mask[1:],col)
  receipt=t.execution.natural_cure[0];before=receipt.pending.before._replace(status=receipt.pending.before.status.at[0,0,3].set(0));badreceipt=receipt._replace(pending=receipt.pending._replace(before=before));bad=t._replace(execution=t.execution._replace(natural_cure=(badreceipt,)));self.assertFalse(bool(nc.owned(bad,0)[0]))
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   self.assertFalse(bool(nc.owned(t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))}),0).any()))
 def test_legacy_missing_and_replayed_source_facts_stay_closed(self):
  t,s,side,old,new,yes=self.component();ctx=self.runtime['direct_hit_context'];r=j.arange(8)
  legacy=ctx._replace(data=ctx.data._replace(baton_pass=ctx.data.baton_pass._replace(natural_cure=None)))
  _,pending=nc.begin(t,s,side,new,self.table,legacy,eligible=yes,baton=~yes);self.assertIsNone(pending)
  # The original dormant healthy-Natural-Cure route remains quiet, whereas
  # legacy metadata alone cannot authorize the newly emitted status cure.
  healthy=s._replace(status=j.zeros_like(s.status),status_counter=j.zeros_like(s.status_counter),sleep_skipped=j.zeros_like(s.sleep_skipped),rest_origin=j.zeros_like(s.rest_origin))
  self.assertEqual(bp._quiet(healthy,side,old,legacy.data.baton_pass,outgoing_dormant=True).tolist(),[True]*8)
  self.assertEqual(bp._quiet(s,side,old,legacy.data.baton_pass,outgoing_dormant=True).tolist(),[False]*6+[True,False])
  _,pending=nc.begin(t,s._replace(happiness=None),side,new,self.table,ctx,eligible=yes,baton=~yes);self.assertIsNone(pending)
  for field,value in [('volatile',1),('ability',self.quiet)]:
   changed=s._replace(**{field:getattr(s,field).at[r,side,old].set(value)})
   tape,pending=nc.begin(t,changed,side,new,self.table,ctx,eligible=yes,baton=~yes)
   after=entry.depart(changed,side,self.table.entry,eligible=yes);out=nc.finish(tape,pending,after)
   self.assertEqual(out.count.tolist(),[0]*8,(field,value))
  tape,pending=nc.begin(t,s,side,new,self.table,ctx,eligible=yes,baton=~yes)
  badmask=pending._replace(entry_mask=~pending.entry_mask);out=nc.finish(tape,badmask,entry.depart(s,side,self.table.entry,eligible=yes))
  self.assertEqual(out.count.tolist(),[0]*8);self.assertTrue(bool(j.all((out.execution.ledger.refusal[:6]&ex.UNKNOWN_SITE)!=0)))
 def test_baton_three_word_install_original_full_both_owners(self):
  from types import SimpleNamespace
  from tempfile import TemporaryDirectory
  from gpu_engine import engine_request_order as ordering,source_provenance
  b=2;r=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);side=r%2
  t,s,_,_,_,_=self.component(b=2,statuses=z+4)
  active=j.zeros((b,2),j.int32);old=z;incoming=z+3;v=self.metadata['vocabulary'];moves=j.broadcast_to(j.array([v['moves'].index(n)for n in self.names],j.int32),(b,2,6,4));pp=j.broadcast_to(j.array([64,24,32,48],j.int32),(b,2,6,4))
  s=s._replace(active=active,turn=z,status=j.zeros_like(s.status).at[r,side,old].set(4),status_counter=j.zeros_like(s.status_counter).at[r,side,old].set(2),ability=j.full_like(s.ability,self.quiet).at[r,side,old].set(self.ability),base_ability=j.full_like(s.base_ability,self.quiet).at[r,side,old].set(self.ability),species=s.species.at[r,side,incoming].set(2),moves=moves,pp=pp,item=j.full_like(s.item,v['items'].index('leftovers')),stats=j.full_like(s.stats,100))
  s=s._replace(types=self.rules.device.species_types[s.species],base_types=self.rules.device.species_types[s.species],boosts=s.boosts.at[r,side,old,2].set(1).at[r,side,old,3].set(1))
  cursor=life.register(t.stream_id,**self.kw);initial=life._tape(cursor,self.producer,64)
  for owner in(0,1):initial=life._reveal(initial,s,z+owner,z-1,z,eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);cursor=initial.cursor
  # BOTH literal prior-public views contain the same explicit status/boost
  # source facts. No accepted flag or historical missing value is supplied.
  ns={k:np.asarray(getattr(self.runtime['public'].device.namespaces,k))for k in full._NAMES};kw=v4.wire_kwargs(self.catalog);name=self.catalog.species.names[int(self.runtime['public'].device.causal.species[1])];incoming_name=self.catalog.species.names[int(self.runtime['public'].device.causal.species[2])];prefixes=[];wire_before=[]
  for row in range(b):
   a=row%2;ids=[f'p{i+1}a: '+name for i in(0,1)];lines=[f'|switch|{ids[i]}|{name}, L80, M|100/100'+(' tox'if i==a else'')for i in(0,1)]+['|turn|1',f'|-boost|{ids[a]}|spa|1',f'|-boost|{ids[a]}|spd|1'];wire,packets=public_wire.project_lines(public_wire.WireState(),lines,**kw);wire_before.append(wire);prefixes.append([e for packet in packets for e in packet.events])
  owners=[]
  for owner in(0,1):
   p=v4.initialize(self.catalog,stream_ids=t.stream_id,owner_sides=z+owner)
   for chunk in inputs.pack_chunks(np.asarray(t.stream_id),prefixes,**ns):p,error=v4.fold_chunk(p,chunk,self.catalog);self.assertFalse(bool(error.any()))
   stream=full.register(t.stream_id,z+owner,compiled=self.runtime['public']);stream=stream._replace(causal=stream.causal._replace(cursor=cursor,sequence=z+5),sequence=p.cursor.sequence,live=j.ones((b,2),bool),species=p.mons.species,status=p.mons.status,hp=p.mons.hp_numerator,maximum=p.mons.hp_denominator);owners.append((stream,p))
  arrays={name:np.asarray(getattr(s,name)).reshape((4,)+getattr(s,name).shape[2:])for name in('species','level','types','stats','moves','pp','item','ability','hidden_power_type','hidden_power_power','gender')};arrays.update(nature=np.zeros((4,6),np.int32),evs=np.zeros((4,6,6),np.int32),metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8));indices=np.array([[0,1],[2,3]],np.int64)
  with TemporaryDirectory()as temp:
   path=Path(temp)/'unit.npz';np.savez_compressed(path,**arrays);pool=source_provenance.load_constructor_pool(path)
  regs=tuple(request._registration(s,pool,indices,t.stream_id,z+i)for i in(0,1));orders=tuple(ordering.initialize_order(t.stream_id,z+i,z+6,engine_binding=j.broadcast_to(self.rules.device.engine_binding,(b,8)))for i in(0,1))
  def requests(s,cursor):return tuple(request.produce_request(self.req,s,o,reg,life.snapshot(cursor),lifecycle_admitted=yes)for o,reg in zip(orders,regs))
  def new_tape(s,cursor,ledger=None):
   tape=life._tape(cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device;ledger=ex.initialize(tape.stream_id,compiled=account)if ledger is None else ledger;tape=ex.begin_on_tape(tape,ledger,account.device,active=yes);return tape._replace(execution=tape.execution._replace(risks=risk.Context(risk.begin_frame_device(risk.initialize_device(tape.stream_id,rt)._replace(frame=ledger.frame),tape,rt,active=yes),rt)))
  def prepare(tape,s,cursor,h,qs,owners,actions):return bp.prepare(tape,s,h,bp.public_proof(s,cursor,tuple(p for _,p in owners),yes,self.helper.device.baton_pass),qs,orders,actions,self.helper.device.baton_pass)
  consume=jax.jit(lambda ledger,frame,stream,p:ex.consume_frame_device(ledger,frame,stream,p,active=yes,tables=self.runtime['execution_compiled'].device,adapter_tables=self.runtime['public'].device,catalog=self.catalog));captures=[]
  def drain(tape,s,old_cursor,owners,label):
   frame=life.LifecycleResult(s,tape.cursor,tape,life.snapshot(old_cursor),life.snapshot(tape.cursor),~yes,z);new=[];ledgers=[]
   for owner,(stream,p)in enumerate(owners):
    ledger,got=consume(tape.execution.ledger,frame,stream,p);captures.append(dict(label=label,owner=owner,raw=tape.words.tolist(),count=tape.count.tolist(),words=got.projection.frames.words.tolist(),valid=got.projection.frames.valid.tolist(),error=got.projection.error.tolist(),stage=ledger.stage.tolist(),refusal=ledger.refusal.tolist()))
    Path('/tmp/pokezero-natural-cure-baton-actual.json').write_text(json.dumps(captures)+'\n')
    self.assertEqual(got.projection.error.tolist(),[0,0],captures[-1]);new.append((got.projection.state,got.public));ledgers.append(ledger)
   return frame,new,ledgers
  # Actual before/request source, including the original one-PP declaration.
  s=s._replace(pp=s.pp.at[r,side,old,0].add(-1));h=bp.empty_history(s,cursor,j.stack(tuple(o.request_to_physical for o in orders),axis=1));qs=requests(s,cursor);tape=prepare(new_tape(s,cursor),s,cursor,h,qs,owners,j.zeros((b,2),j.int32));tape=life.begin(tape,s,yes,~yes);move=z+self.move;tape=public.move(tape,side,old,z,move,eligible=yes);tape,p=bp.begin_start(tape,s,side,move,move,self.runtime['direct_hit_context'],acts=yes,eligible=yes);self.assertEqual(p.valid.tolist(),[True,True])
  s=s._replace(baton_pass=s.baton_pass.at[r,side].set(True));tape=bp.finish_start(tape,p,s,yes,yes);s=s._replace(forced_switch=s.baton_pass,pending_residual=yes,turn=z+1);tape=bp.finalize_request(tape,s);frame,owners,ledgers=drain(tape,s,cursor,owners,'START');qs=requests(s,tape.cursor);history=bp.mature(tape.execution.baton_transaction.history,frame,tuple(ledgers),tuple(p for _,p in owners),qs);self.assertEqual(history.ready.tolist(),[True,True]);joined,_=ex.join_consumed_owners(*ledgers,tables=self.runtime['execution_compiled'].device)
  for row in range(b):
   a=row%2;_,packets=public_wire.project_lines(wire_before[row],['|','|t:|1',f'|move|p{a+1}a: {name}|Baton Pass|p{a+1}a: {name}'],**kw)
   actual=np.asarray(captures[0]['words']).transpose(1,0,2,3)[row][np.asarray(captures[0]['valid']).transpose(1,0,2)[row]];np.testing.assert_array_equal(actual,np.array([e.words()for packet in packets for e in packet.events],np.int32))
  # Actual source copy/depart/install, with ONE Site39 three-word ownership.
  old_cursor=tape.cursor;actions=j.zeros((b,2),j.int32).at[r,side].set(7);tape=prepare(new_tape(s,old_cursor,joined),s,old_cursor,history,qs,owners,actions);self.assertEqual(tape.execution.baton_transaction.request_ok.tolist(),[True,True]);tape=life.begin(tape,s,yes,~yes);tape,p=bp.before_copy(tape,s,side,incoming,self.runtime['direct_hit_context'],eligible=yes);self.assertEqual(p.valid.tolist(),[True,True]);s=baton.copy_to_recipient(s,side,incoming,eligible=yes);tape=bp.after_copy(tape,p,s)
  tape,npending=nc.begin(tape,s,side,incoming,self.table,self.runtime['direct_hit_context'],eligible=yes,baton=yes);self.assertEqual(nc.prospective(npending).tolist(),[True,True]);s=entry.depart(s,side,self.table.entry,eligible=yes);tape=nc.finish(tape,npending,s);self.assertEqual(tape.count.tolist(),[3,3]);self.assertEqual(nc.matches_baton(tape,tape.execution.baton_transaction.record,stage=1).tolist(),[True,True]);s=s._replace(active=s.active.at[r,side].set(incoming),baton_pass=j.zeros_like(s.baton_pass),forced_switch=j.zeros_like(s.forced_switch),active_turns=s.active_turns.at[r,side].set(0));tape=life.installed(tape,s,side,old,incoming,eligible=yes,replacement=yes,baton=yes);tape=bp.finish_install(tape,s,side,eligible=yes)
  Path('/tmp/pokezero-natural-cure-baton-install-claims.json').write_text(json.dumps(dict(raw=tape.words.tolist(),count=tape.count.tolist(),stage=tape.execution.ledger.stage.tolist(),refusal=tape.execution.ledger.refusal.tolist(),parent_owned=bp.owned(tape,tape.count-1,bp.INSTALL).tolist(),cure_owned=nc.owned(tape,2).tolist(),established=tape.cursor.established.tolist()))+'\n')
  self.assertEqual(bp.owned(tape,tape.count-1,bp.INSTALL).tolist(),[True,True]);self.assertEqual(nc.owned(tape,2).tolist(),[True,True]);self.assertEqual(tape.execution.ledger.site[r,tape.execution.baton_transaction.record.slot].tolist(),[39,39]);self.assertEqual((tape.execution.ledger.end-tape.execution.ledger.start)[r,tape.execution.baton_transaction.record.slot].tolist(),[3,3])
  frame,new,closed=drain(tape,s,old_cursor,owners,'INSTALL');self.assertEqual(ex.join_consumed_owners(*closed,tables=self.runtime['execution_compiled'].device)[1].tolist(),[True,True])
  exact=[]
  for owner in(0,1):
   record=captures[2+owner]
   for row in range(b):
    a=row%2;wire,_=public_wire.project_lines(wire_before[row],['|','|t:|1',f'|move|p{a+1}a: {name}|Baton Pass|p{a+1}a: {name}'],**kw);_,packets=public_wire.project_lines(wire,['|','|t:|1',f'|-curestatus|p{a+1}a: {name}|tox|[from] ability: Natural Cure',f'|switch|p{a+1}a: {incoming_name}|{incoming_name}, L80, M|100/100'],**kw);expected=np.array([e.words()for packet in packets for e in packet.events],np.int32);actual=np.asarray(record['words']).transpose(1,0,2,3)[row][np.asarray(record['valid']).transpose(1,0,2)[row]];exact.append(dict(owner=owner,row=row,actual=actual.tolist(),expected=expected.tolist()));Path('/tmp/pokezero-natural-cure-baton-word-comparisons.json').write_text(json.dumps(exact,sort_keys=True)+'\n');np.testing.assert_array_equal(actual,expected)
  capture_path(self,'natural-cure-baton-original-full-compact.json').write_text(json.dumps(exact,sort_keys=True)+'\n')
