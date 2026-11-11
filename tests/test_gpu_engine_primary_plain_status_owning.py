"""Prospective literal primary PAR/BRN input; no scientific roots/models."""
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_primary_plain_status_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk,engine_status_callbacks as status
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers,engine_status_residual_callbacks as residual
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,source_provenance,mechanics as m
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from test_gpu_engine_gameplay_stream_comparison import state as literal_state
from _showdown_root import showdown_root
import test_gpu_engine_bulk_up_owning as fixtures

class PrimaryPlainStatusTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.BulkUpOwningTests.setUpClass();cls.f=fixtures.BulkUpOwningTests;cls.metadata=deepcopy(cls.f.metadata);dex=load_showdown_dex_cached(showdown_root());moves=dict(cls.f.dex.moves);cls.names=('willowisp','stunspore','surf','thunderwave');labels=set(cls.f.catalog.vocabulary.tokens)
  for name in cls.names[:2]:
   info=dex.move_info(name);moves[name]=info
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   at=cls.metadata['vocabulary']['moves'].index(name);cls.metadata['moves'][at]=dict(id=name,base_power=0,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,category='Status',pp=info.pp,accuracy=75,priority=0,flags={'protect':1,'reflectable':1});labels|={'move:'+name,'belief:possible_move:'+name,'move_effect:'+info.status}
  cls.dex=ShowdownDex(moves,cls.f.dex.species,cls.f.dex.type_chart);variants={n:Gen3RandbatSpeciesUniverse(n,80,(Gen3RandbatVariant('plain-'+n,'authored',n,'authored',80,cls.names,'Pressure','Leftovers'),))for n in cls.metadata['vocabulary']['species'][1:]};source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-primary-plain-source'),variants)
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=cls.f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False));cls.rules=request.load_source_rules(showdown_root(),cls.metadata);cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer)
  cls.facts=own.compile_own_facts(cls.req,cls.catalog);cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(cls.req,cls.facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',status_residual_compiled=residual.compile_status_residual(cls.metadata,engine_catalog=cls.producer),status_residual_mode='kernel-event',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32));cls.fold=staticmethod(jax.jit(lambda s,f,p:full.project_and_fold(s,f,cls.runtime['public'].device,p,cls.catalog)))
 def test_original_lexical_scope_default_and_legacy_disjoint(self):
  from gpu_engine import engine_source_frame_witness as witness
  witness._source();self.assertEqual(len(witness.SOURCE_PINS),88);d=self.helper.device.primary_plain_status
  for name,code in (('willowisp',m.BRN),('stunspore',m.PAR)):
   at=self.metadata['vocabulary']['moves'].index(name);self.assertTrue(bool(d.candidates[at]));self.assertEqual(int(d.status[at]),code);self.assertEqual(int(self.runtime['kernel_risk_compiled'].device.selected_risk[at]),0)
  for name in ('thunderwave','sleeppowder','bulkup'):self.assertFalse(bool(d.candidates[self.metadata['vocabulary']['moves'].index(name)]))
  self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.primary_plain_status)
  # Prove the actual plain-status parent, before the parallel Color Change merge.
  old=ast.parse(subprocess.check_output(['git','show','ec5878a1:gpu_engine/transition.py'],text=True));after=ast.parse(subprocess.check_output(['git','show','bad7cd59:gpu_engine/transition.py'],text=True));baseline=dump_default_program(old);self.assertEqual(baseline,dump_default_program(after));assert_hp_rng_mutations_detected(self,after,baseline)
  after=ast.parse(Path('gpu_engine/transition.py').read_text());integrated=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True))
  self.assertEqual(ast.dump(after),ast.dump(integrated));assert_hp_rng_mutations_detected(self,after,dump_default_program(integrated))
  script="const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),out=[];for(const id of ['willowisp','stunspore']){let m=d.moves.get(id),p={toString:()=> 'p2a: Target'},lines=[];d.conditions.get(m.status).onStart.call({add:(...v)=>lines.push(v.map(String).join('|'))},p,{toString:()=> 'p1a: Actor'},m);out.push(lines);}process.stdout.write(JSON.stringify(out));"
  self.assertEqual(json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True)),[['-status|p2a: Target|brn'],['-status|p2a: Target|par']])
 def component(self,*,consume=True):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);no=~yes;z=j.zeros(b,j.int32);side=rows%2;s=jax.tree.map(j.asarray,literal_state(b));active=j.stack((z+3,z+4),axis=1);s=s._replace(active=active,species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),ability=j.zeros_like(s.ability),base_ability=j.zeros_like(s.base_ability),gender=j.ones_like(s.gender));cursor=life.register(rows+45001,**self.kw);initial=life._tape(cursor,self.producer,64)
  for st in (0,1):initial=life._reveal(initial,s,z+st,z-1,active[:,st],eligible=yes,replacement=no,baton=no,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);frame=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(cursor),life.snapshot(initial.cursor),no,z);consumers=[]
  for owner in ((0,1)if consume else ()):
   stream=full.register(cursor.stream_id,z+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=cursor.stream_id,owner_sides=z+owner);r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error.tolist(),[0]*b);consumers.append((r.projection.state,r.public))
  tape=life._tape(initial.cursor,self.producer,64);account=self.runtime['execution_compiled'];r=self.runtime['kernel_risk_compiled'];tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,r.device),tape,r.device,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,r.device)),refusal=j.array([1]*7+[257],j.int32))
  move=j.where(rows%2==0,self.metadata['vocabulary']['moves'].index('willowisp'),self.metadata['vocabulary']['moves'].index('stunspore'));physical=active[rows,side];target=active[rows,1-side];eligible=yes.at[6].set(False)
  s=s._replace(status_counter=s.status_counter.at[2,1-int(side[2]),int(target[2])].set(1),status=s.status.at[3,1-int(side[3]),int(target[3])].set(m.BRN));tape=public.move(tape,side,physical,target,move,eligible=yes);tape,p=cb.begin(tape,s,side,move,self.runtime['direct_hit_context'],self.table,eligible=eligible)
  old=s.status[rows,1-side,target];oc=s.status_counter[rows,1-side,target];new=self.helper.device.primary_plain_status.status[move];new=new.at[3].set(old[3]);applied=yes.at[4].set(False).at[5].set(False);applied &= eligible&(old==m.NONE);new=j.where(applied,new,old);hit=yes.at[4].set(False);counter=j.where(applied,0,oc)
  tape=status.changed(tape,1-side,target,old,new,status.PRIMARY,eligible=applied)
  w=cb.Attempt(hit,hit,no,no,no,yes,applied,no,no,no,hit,old,new,oc,counter);tape,captured=cb.finish(tape,p,w);return s,initial.cursor,consumers,tape,captured
 def test_mixed_counters_nochange_inactive_sticky_physical_and_both_full(self):
  s,cursor,consumers,tape,captured=self.component();self.assertEqual(cb.owned_pair(tape,2).tolist(),[True,True,False,False,False,False,False,True]);self.assertEqual(cb.captured_matches(tape,captured).tolist(),[True,True,False,False,False,False,False,True]);self.assertEqual(int(tape.refusal[7]),257);self.assertEqual(int(tape.refusal[6]),1);self.assertTrue(bool(j.all((tape.refusal[2:6]&256)!=0)))
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=tape._replace(**{name:getattr(tape,name).at[0].add(j.uint32(1))});self.assertFalse(bool(cb.owned_pair(bad,2).any()),name)
   with self.assertRaises(ValueError):cb.owned_pair(tape._replace(**{name:j.broadcast_to(getattr(tape,name),(8,8))}),2)
  for name in ('turn','phase','generation'):
   bad=tape._replace(cursor=tape.cursor._replace(**{name:getattr(tape.cursor,name)+1}));self.assertFalse(bool(cb.captured_matches(bad,captured).any()),name)
  tape=tape._replace(count=j.where(j.arange(8)<2,tape.count,0));frame=life.LifecycleResult(s,tape.cursor,tape,life.snapshot(cursor),life.snapshot(tape.cursor),j.zeros(8,bool),j.zeros(8,j.int32))
  for owner,(stream,pub)in enumerate(consumers):
   r=self.fold(stream,frame,pub);self.assertEqual(r.projection.error[:2].tolist(),[0,0])
   for row in (0,1):
    names={st:self.catalog.species.names[st+1]for st in (0,1)};ids={st:f'p{st+1}a: '+names[st]for st in (0,1)};who=row%2;move='Will-O-Wisp'if row==0 else 'Stun Spore';cond='brn'if row==0 else 'par'
    wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ids[st]}|{names[st]}, L80, M|100/100'for st in (0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));_,packets=public_wire.project_lines(wire,[f'|move|{ids[who]}|{move}|{ids[1-who]}',f'|-status|{ids[1-who]}|{cond}'],**v4.wire_kwargs(self.catalog));want=np.asarray([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(r.projection.frames.words[:,row])[np.asarray(r.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,want)
 def test_pending_old_status_joins_actual_literal_span(self):
  _,_,_,tape,captured=self.component(consume=False)
  self.assertEqual(cb.captured_matches(tape,captured)[:2].tolist(),[True,True])
  bad=captured._replace(pending=captured.pending._replace(old_status=captured.pending.old_status.at[0].set(m.BRN)))
  self.assertEqual(cb.captured_matches(tape,bad)[:2].tolist(),[False,True])
 def test_actual_original_frame_both_owners_witness_encoder_and_physics(self):
  arrays=self.f.f.f.f.base().arrays('pressure','pressure');arrays['moves'][:]=np.array([self.metadata['vocabulary']['moves'].index(n)for n in self.names]);arrays['pp'][:]=[24,48,24,32];arrays['item'][:]=self.metadata['vocabulary']['items'].index('leftovers');arrays['stats'][:,:,0]=300;arrays['stats'][:,:,3]=50;arrays['stats'][:,:,4]=200;arrays['stats'][0,:,5]=150;arrays['stats'][1,:,5]=220
  # Original literal species/type rows with no status-immunity or cure source.
  for st,name in enumerate(('pikachu','pidgeot')):
   arrays['species'][st,0]=self.metadata['vocabulary']['species'].index(name);types=self.dex.species_info(name).types;arrays['types'][st,0]=[self.metadata['vocabulary']['types'].index(t.lower())for t in types]+[0]*(2-len(types))
  with TemporaryDirectory()as tmp:
   p=Path(tmp)/'literal.npz';np.savez_compressed(p,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8)));pool=source_provenance.load_constructor_pool(p);session=driver.construct(pool,np.tile([[0,1]],(2,1)).astype(np.int64),prepared=self.prepared,stream_ids=j.array([46001,46002],j.int32),seed=31)
  self.assertEqual(session.state.core.admitted.tolist(),[True,True]);self.assertEqual(session.state.core.weather_profile.qualified_birth.tolist(),[True,True]);before=session.state;actions=j.array([[0,2],[2,1]],j.int32);draws=j.zeros((2,2,12),j.int32).at[:,:,4].set(1).at[:,:,2].set(99).at[:,:,3].set(99);chance=tr.ChanceInputs(draws,*[j.zeros((2,2),j.int32)for _ in range(4)],j.zeros(2,j.int32));frames=[];original=self.runtime['frame']
  def capture(*a,**kw):r=original(*a,**kw);frames.append(r);return r
  self.runtime['frame']=capture
  try:result=driver.advance(session,actions,chance_inputs=chance)
  finally:self.runtime['frame']=original
  frame=frames[-1];plain=jax.jit(lambda state,actions:tr.step_with_chance(state,actions,self.table,chance).state)(before.core.engine,actions)
  for a,z in zip(jax.tree.leaves(frame.state),jax.tree.leaves(plain),strict=True):np.testing.assert_array_equal(a,z)
  self.assertEqual(result.state.core.admitted.tolist(),[True,True],json.dumps(dict(raw=frame.callbacks.words.tolist(),source=frame.callbacks.refusal.tolist(),risk=result.state.core.risks.risk.tolist(),ledger=result.state.core.execution.refusal.tolist(),witness=result.state.core.frame_input_report.refusal.tolist(),expected=result.state.core.frame_input_report.expected_cutpoints.tolist(),actual=result.state.core.frame_input_report.observed_cutpoints.tolist(),owners=[o.reason.tolist()for o in result.owners])))
  for row,code in ((0,m.BRN),(1,m.PAR)):
   raw=np.asarray(frame.callbacks.words[row,:int(frame.callbacks.count[row])]);pairs=raw[raw[:,0]==26];self.assertEqual(len(pairs),1);self.assertEqual(pairs[0,7:10].tolist(),[0,code,0])
  ledger=result.state.core.execution;used=ledger.site==int(ex.Site.PRIMARY_PLAIN_STATUS);self.assertTrue(bool(j.all(j.where(used,ledger.stage==3,True))))
  for owner in result.owners:self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True]);self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True]);self.assertFalse(bool(owner.B4_qualified.any()))

if __name__=='__main__':unittest.main()
