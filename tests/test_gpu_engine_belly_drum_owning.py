"""Quarantined literal Belly Drum components; no scientific games/models."""
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
from gpu_engine import engine_belly_drum_callbacks as cb,engine_direct_hit_controls as direct
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_full_v4_event_device as full,engine_leftovers_callbacks as leftovers
from gpu_engine import transition as tr,public_belief,v4_encoder as v4,public_wire,source_provenance
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from test_gpu_engine_gameplay_stream_comparison import state as literal_state
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixture

class BellyDrumOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.NoDamageOwningTests.setUpClass();cls.f=fixture.NoDamageOwningTests;cls.metadata=deepcopy(cls.f.metadata)
  dex=load_showdown_dex_cached(showdown_root());moves=dict(cls.f.catalog.dex.moves);cls.names=('bellydrum','raindance','sunnyday','hail');info=dex.move_info('bellydrum');moves['bellydrum']=info
  if 'bellydrum' not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append('bellydrum');cls.metadata['moves'].append(None)
  cls.move=cls.metadata['vocabulary']['moves'].index('bellydrum');cls.metadata['moves'][cls.move]=dict(id='bellydrum',base_power=0,type=cls.metadata['vocabulary']['types'].index('normal'),source_type='Normal',category='Status',pp=10,accuracy=True,priority=0,flags={'snatch':1})
  cls.dex=ShowdownDex(moves,cls.f.catalog.dex.species,cls.f.catalog.dex.type_chart)
  variants={n:Gen3RandbatSpeciesUniverse(n,80,(Gen3RandbatVariant('drum-'+n,'authored',n,'authored',80,cls.names,'Pressure','Leftovers'),))for n in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-belly-drum-source'),variants)
  labels=set(cls.f.catalog.vocabulary.tokens)|{'move:bellydrum','belief:possible_move:bellydrum'}
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=cls.f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  cls.rules=request.load_source_rules(showdown_root(),cls.metadata);cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer)
  cls.facts=own.compile_own_facts(cls.req,cls.catalog);cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
  cls.prepared=driver.prepare(cls.req,cls.facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32))

 def test_original_literal_source_order_default_program_and_static_scope(self):
  from gpu_engine import engine_source_frame_witness as witness
  witness._source();self.assertTrue(bool(self.helper.device.belly_drum.candidates[self.move]));self.assertEqual(int(self.runtime['kernel_risk_compiled'].device.selected_risk[self.move]),0)
  self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.belly_drum)
  old=ast.parse(subprocess.check_output(['git','show','977d524d:gpu_engine/transition.py'],text=True));after=ast.parse(subprocess.check_output(['git','show','0087033e:gpu_engine/transition.py'],text=True));baseline=dump_default_program(old)
  # The immutable Belly Drum delta precedes the later pinch observer.
  self.assertEqual(dump_default_program(after),baseline);assert_hp_rng_mutations_detected(self,after,baseline)
  current=ast.parse(Path('gpu_engine/transition.py').read_text());integrated=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True))
  self.assertEqual(ast.dump(current),ast.dump(integrated));assert_hp_rng_mutations_detected(self,current,dump_default_program(integrated))
  script=r'''const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),B=require(process.argv[1]+'/dist/sim/battle.js').Battle,P=require(process.argv[1]+'/dist/sim/pokemon.js').Pokemon,m=d.moves.get('bellydrum'),out=[];
for(const [hp,maxhp,atk]of [[201,201,0],[101,201,-6],[100,201,0],[201,201,6],[1,1,0]]){
const lines=[],p={hp,maxhp,isActive:true,boosts:{atk},battle:{clampIntRange:(v,a,z)=>Math.max(a,Math.min(z,v))},getCappedBoost:P.prototype.getCappedBoost,boostBy:P.prototype.boostBy,
toString:()=> 'p2a: Actor',getHealth(){return this.hp+'/'+this.maxhp},damage(n){const old=this.hp;this.hp-=Math.min(this.hp,n);return old-this.hp}};
const b={gen:3,event:{target:p,source:p},effect:m,clampIntRange:(v,a,z=Infinity)=>Math.max(a,Math.min(z,Math.floor(v))),dex:d,
runEvent:(id,t,s,e,v)=>v,add:(...v)=>lines.push(v.map(x=>typeof x==='function'?x.call(p):String(x)).join('|')),boost:B.prototype.boost,directDamage:B.prototype.directDamage};
const result=m.onHit.call(b,p);out.push({lines,hp:p.hp,atk:p.boosts.atk,failed:result===false});}process.stdout.write(JSON.stringify(out));'''
  values=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True))
  self.assertEqual(values[0]['lines'],['-damage|p2a: Actor|101/201','-setboost|p2a: Actor|atk|6|[from] move: Belly Drum'])
  self.assertEqual(values[1]['hp'],1);self.assertEqual(values[1]['atk'],6)
  self.assertTrue(all(v['failed']and not v['lines']for v in values[2:]))
  cat=self.catalog;name=cat.species.names[1];ident='p1a: '+name
  prior,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ident}|{name}, L80, M|100/100',f'|switch|p2a: {name}|{name}, L80, M|100/100','|turn|1'],**v4.wire_kwargs(cat))
  _,packet=public_wire.project_line(prior,f'|-setboost|{ident}|atk|6|[from] move: Belly Drum',**v4.wire_kwargs(cat));expected=np.array([e.words()for e in packet.events],np.int32)
  key=int(self.runtime['public'].device.belly_drum_setboost);np.testing.assert_array_equal(np.asarray(self.runtime['public'].device.templates[key])[:len(expected)],expected)

 def component(self):
  b=8;rows=j.arange(b);yes=j.ones(b,bool);z=j.zeros(b,j.int32);side=rows%2
  s=jax.tree.map(j.asarray,literal_state(b));active=j.stack((z+3,z+4),axis=1);s=s._replace(active=active,species=j.broadcast_to(j.array([1,2],j.int32)[None,:,None],(b,2,6)),ability=j.zeros_like(s.ability),base_ability=j.zeros_like(s.base_ability))
  old=j.array([201,101,100,201,1,201,201,201],j.int32);maximum=j.array([201,201,201,201,1,201,201,201],j.int32);atk=j.array([0,-6,0,6,0,0,0,0],j.int32);where=active[rows,side]
  s=s._replace(hp=s.hp.at[rows,side,where].set(old),max_hp=s.max_hp.at[rows,side,where].set(maximum),boosts=s.boosts.at[rows,side,where,0].set(atk))
  cursor=life.register(rows+59001,**self.kw);tape=life._tape(cursor,self.producer,64)
  for seat in (0,1):tape=life._reveal(tape,s,z+seat,z-1,active[:,seat],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  cursor=tape.cursor._replace(initialized=yes,turn=z+1);tape=life._tape(cursor,self.producer,64);account=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device
  tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes)
  carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,rt),tape,rt,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,rt)))
  requested=yes.at[5].set(False);move=(z+self.move).at[6].set(self.metadata['vocabulary']['moves'].index('raindance'))
  tape=public.move(tape,side,where,active[rows,1-side],move,eligible=requested)
  tape,p=cb.begin(tape,s,side,move,old,s.boosts[rows,side,where],self.runtime['direct_hit_context'],self.table.effects,eligible=requested)
  drum=requested&(move==self.move)&(maximum>1)&(old*2>maximum)&(atk<6)
  after=s._replace(hp=s.hp.at[rows,side,where].set(old-j.where(drum,maximum//2,0)),boosts=s.boosts.at[rows,side,where,0].set(j.where(drum,6,atk)))
  tape,captured=cb.finish(tape,p,after,yes)
  return tape,captured,side,after

 def test_same_write_mixed_guards_prior_bits_and_inactive_rows(self):
  tape,captured,side,s=self.component();self.assertEqual(cb.captured_matches(tape,captured).tolist(),[True,True,False,False,False,False,False,True]);self.assertEqual(tape.count.tolist(),[3,3,1,1,1,0,1,3])
  observed=risk.branch_on_tape(tape,risk.Branch.BELLY_DRUM,captured.pending.requested,active=j.ones(8,bool),captured_belly_drum=captured)
  self.assertEqual((observed.execution.risks.carry.risk&risk.SELF_EFFECT).tolist(),[0,0,2,2,2,0,0,0])
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=tape._replace(**{name:getattr(tape,name).at[0].add(j.uint32(1))});self.assertFalse(bool(cb.pair(bad,1)[2].any()),name)
   with self.assertRaises(ValueError):cb.pair(tape._replace(**{name:j.broadcast_to(getattr(tape,name),(8,8))}),1)
  for label,bad in (('prefix',tape._replace(words=tape.words.at[0,0,5].set(0))),('status',tape._replace(words=tape.words.at[0,1,8].set(6))),('claim',tape._replace(execution=tape.execution._replace(ledger=tape.execution.ledger._replace(claimed=tape.execution.ledger.claimed.at[0,2].set(0)))))):
   with self.subTest(label=label):self.assertFalse(bool(cb.pair(bad,1)[2][0]))
  prior=tape._replace(refusal=tape.refusal.at[0].set(65));got=risk.branch_on_tape(prior,risk.Branch.BELLY_DRUM,captured.pending.requested,active=j.ones(8,bool),captured_belly_drum=captured);self.assertEqual(int(got.refusal[0])&65,65)


 def test_original_both_full_pair_all16_words_and_fold(self):
  tape,captured,side,after=self.component();rows=j.arange(8);physical=after.active[rows,side]
  before=after._replace(hp=after.hp.at[rows,side,physical].set(captured.write.old_hp),boosts=after.boosts.at[rows,side,physical,0].set(captured.write.old_atk))
  zero=j.zeros(8,j.int32);yes=j.ones(8,bool);start=life.register(tape.stream_id,**self.kw);initial=life._tape(start,self.producer,64)
  for seat in (0,1):initial=life._reveal(initial,before,zero+seat,zero-1,before.active[:,seat],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes)
  frame0=life.LifecycleResult(before,initial.cursor,initial,life.snapshot(start),life.snapshot(initial.cursor),~yes,zero)
  frame=life.LifecycleResult(after,tape.cursor,tape,life.snapshot(tape.cursor),life.snapshot(tape.cursor),~yes,zero)
  fold=jax.jit(lambda st,fr,p:full.project_and_fold(st,fr,self.runtime['public'].device,p,self.catalog))
  for owner in (0,1):
   stream=full.register(tape.stream_id,zero+owner,compiled=self.runtime['public']);pub=v4.initialize(self.catalog,stream_ids=tape.stream_id,owner_sides=zero+owner)
   init=fold(stream,frame0,pub);self.assertEqual(init.projection.error[:2].tolist(),[0,0])
   got=fold(init.projection.state,frame,init.public);self.assertEqual(got.projection.error[:2].tolist(),[0,0])
   for row in (0,1):
    lines=[];ident=[]
    for seat in (0,1):
     at=int(before.active[row,seat]);species=int(before.species[row,seat,at]);name=self.catalog.species.names[int(self.runtime['public'].device.causal.species[species])];ident.append(f'p{seat+1}a: '+name)
     hp=int(before.hp[row,seat,at]);maximum=int(before.max_hp[row,seat,at]);visible=hp if owner==seat else (100*hp+maximum-1)//maximum;den=maximum if owner==seat else 100
     lines.append(f'|switch|{ident[seat]}|{name}, L80, M|{visible}/{den}')
    lines.append('|turn|1');wire,_=public_wire.project_lines(public_wire.WireState(),lines,**v4.wire_kwargs(self.catalog));actor=row
    hp=int(captured.write.new_hp[row]);maximum=int(captured.write.maximum[row]);visible=hp if owner==actor else (100*hp+maximum-1)//maximum;den=maximum if owner==actor else 100
    _,packets=public_wire.project_lines(wire,[f'|move|{ident[actor]}|Belly Drum|{ident[actor]}',f'|-damage|{ident[actor]}|{visible}/{den}',f'|-setboost|{ident[actor]}|atk|6|[from] move: Belly Drum'],**v4.wire_kwargs(self.catalog))
    expected=np.asarray([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])]
    np.testing.assert_array_equal(actual,expected)

 def test_actual_original_frame_both_owners_witness_encoder_and_physics(self):
  arrays=self.f.f.base().arrays('pressure','pressure');arrays['moves'][:]=np.array([self.metadata['vocabulary']['moves'].index(n)for n in self.names]);arrays['pp'][:]=[16,8,8,16];arrays['item'][:]=0
  # Preserve all six distinct original species and canonical constructor types.
  for team in (0,1):
   self.assertEqual(len(set(map(int,arrays['species'][team]))),6)
   for physical,species in enumerate(arrays['species'][team]):
    info=self.dex.species_info(self.metadata['vocabulary']['species'][int(species)]);expected=[self.metadata['vocabulary']['types'].index(t.lower())for t in info.types]+[0]*(2-len(info.types))
    self.assertEqual(arrays['types'][team,physical].tolist(),expected)
  with TemporaryDirectory()as tmp:
   p=Path(tmp)/'literal.npz';np.savez_compressed(p,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8)));pool=source_provenance.load_constructor_pool(p)
   session=driver.construct(pool,np.tile([[0,1]],(2,1)).astype(np.int64),prepared=self.prepared,stream_ids=j.array([60001,60002],j.int32),seed=31)
  self.assertEqual(session.state.core.admitted.tolist(),[True,True]);self.assertTrue(all(bool(o.encoded.observation.admitted.all())for o in session.result.owners));before=session.state
  actions=j.array([[0,1],[1,0]],j.int32);draws=j.zeros((2,2,12),j.int32).at[:,:,4].set(1);chance=tr.ChanceInputs(draws,*[j.zeros((2,2),j.int32)for _ in range(4)],j.zeros(2,j.int32));frames=[];original=self.runtime['frame']
  def capture(*a,**kw):f=original(*a,**kw);frames.append(f);return f
  self.runtime['frame']=capture
  try:result=driver.advance(session,actions,chance_inputs=chance)
  finally:self.runtime['frame']=original
  frame=frames[-1];plain=jax.jit(lambda state,actions:tr.step_with_chance(state,actions,self.table,chance).state)(before.core.engine,actions)
  for a,z in zip(jax.tree.leaves(frame.state),jax.tree.leaves(plain),strict=True):np.testing.assert_array_equal(a,z)
  diagnostic=dict(raw=[np.asarray(frame.callbacks.words[r,:int(frame.callbacks.count[r])]).tolist()for r in (0,1)],source=frame.callbacks.refusal.tolist(),risk=result.state.core.risks.risk.tolist(),ledger=result.state.core.execution.refusal.tolist(),witness=result.state.core.frame_input_report.refusal.tolist(),expected=result.state.core.frame_input_report.expected_cutpoints.tolist(),actual=result.state.core.frame_input_report.observed_cutpoints.tolist(),owners=[o.reason.tolist()for o in result.owners],full_words=np.asarray(result.state.core.frame_input_report.owner_full_words).tolist(),full_valid=np.asarray(result.state.core.frame_input_report.owner_full_valid).tolist())
  Path('/tmp/pokezero-belly-drum-actual-capture.json').write_text(json.dumps(diagnostic,separators=(',',':'))+'\n')
  self.assertEqual(result.state.core.admitted.tolist(),[True,True],json.dumps({k:v for k,v in diagnostic.items()if not k.startswith('full_')}))
  for row,side in ((0,0),(1,1)):
   words=np.asarray(diagnostic['raw'][row]);pair=words[words[:,0]==cb.BELLY_DRUM];self.assertEqual(pair[:,4].tolist(),[0,1]);self.assertEqual(int(pair[1,6]),6)
  ledger=result.state.core.execution;used=ledger.site==int(ex.Site.BELLY_DRUM);self.assertTrue(bool(j.all(j.where(used,ledger.stage==3,True))))
  for owner in result.owners:self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True]);self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True]);self.assertFalse(bool(owner.B4_qualified.any()))
