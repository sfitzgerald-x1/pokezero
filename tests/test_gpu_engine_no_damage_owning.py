"""Prospective literal no-damage target input components, no scientific roots."""
from copy import deepcopy
from dataclasses import replace
import ast,json,os,subprocess,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.category_vocab import CategoryVocabulary
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_no_damage_callbacks as nd,engine_direct_hit_controls as hit,engine_kernel_frame_risk as risk
from gpu_engine import engine_execution_completeness as ex,engine_lifecycle_callbacks as life,engine_public_events as public
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_leftovers_callbacks as leftovers,engine_full_v4_event_device as full
from gpu_engine import public_belief,v4_encoder as v4,public_wire,transition as tr,source_provenance,mechanics as m,absorption
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_sleep_powder_runtime as fixtures

class NoDamageOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.SleepPowderRuntimeTests.setUpClass();cls.f=fixtures.SleepPowderRuntimeTests;cls.metadata=deepcopy(cls.f.metadata)
  dex=load_showdown_dex_cached(showdown_root());cls.names=('earthquake','hydropump','flamethrower','thunderbolt');moves=dict(cls.f.catalog.dex.moves)
  for name in cls.names:
   info=dex.move_info(name)
   if name not in cls.metadata['vocabulary']['moves']:cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
   at=cls.metadata['vocabulary']['moves'].index(name);moves[name]=info
   cls.metadata['moves'][at]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,
       category=info.category,pp=info.pp,accuracy=int(info.accuracy),priority=info.priority,flags={'protect':1})
  # Source-aware compilers inspect the entire installed move domain, including
  # inherited aliases. Populate those fixture rows from the pinned Gen3 Dex.
  installed=cls.metadata['vocabulary']['moves']
  script="const d=require(process.argv[1]).Dex.mod('gen3');console.log(JSON.stringify(JSON.parse(process.argv[2]).map(n=>{const m=d.moves.get(n);return {valid:m.exists&&m.gen<=3,flags:m.flags,accuracy:m.accuracy}})))"
  source_rows=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root()/'dist/sim/dex.js'),json.dumps(installed)],text=True))
  for i,(name,row) in enumerate(zip(installed,source_rows,strict=True)):
   if i==0:continue
   if not row['valid']:
    cls.metadata['moves'][i].setdefault('flags',{})['failencore']=1
    continue
   info=dex.move_info(name)
   if info.type.lower() not in cls.metadata['vocabulary']['types']:continue
   moves[name]=info
   cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),source_type=info.type,
       category=info.category,pp=info.pp,accuracy=row['accuracy'],priority=info.priority,flags=row['flags'])
  for name in ('waterabsorb','voltabsorb','flashfire'):
   if name not in cls.metadata['vocabulary']['abilities']:cls.metadata['vocabulary']['abilities'].append(name)
  species=dict(cls.f.catalog.dex.species)
  for name in ('pidgeot','vaporeon','houndoom','jolteon'):
   cls.metadata['vocabulary']['species'].append(name);species[name]=dex.species_info(name)
  types=cls.metadata['vocabulary']['types'][1:]
  cls.metadata['type_chart']=[[dex.effectiveness(a,(d,))for d in types]for a in types]
  cls.dex=ShowdownDex(moves,species,dex.type_chart)
  abilities=('Pressure','Water Absorb','Volt Absorb','Flash Fire')
  variants={name:Gen3RandbatSpeciesUniverse(name,80,tuple(Gen3RandbatVariant(name+str(i),'authored',name,'authored',80,cls.names,abi,'Leftovers')for i,abi in enumerate(abilities)))for name in cls.metadata['vocabulary']['species'][1:]}
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-no-damage-source'),variants)
  labels=set(cls.f.catalog.vocabulary.tokens)
  for name in cls.names:labels|={'move:'+name,'belief:possible_move:'+name}
  for name in ('waterabsorb','voltabsorb','flashfire'):labels|={'ability:'+name,'belief:possible_ability:'+name}
  for name in ('pidgeot','vaporeon','houndoom','jolteon'):labels.add('species:'+name)
  cls.catalog=v4.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=cls.f.catalog.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
  rules=request.load_source_rules(showdown_root(),cls.metadata);cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=cls.producer)
  facts=own.compile_own_facts(cls.req,cls.catalog);cls.helper=hit.compile_direct_hit_controls(engine_catalog=cls.producer)
  cls.prepared=driver.prepare(cls.req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',
      leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
  cls.runtime=driver._checked(cls.prepared);cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32))
 def same(self,a,b):
  for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):np.testing.assert_array_equal(x,y)
 def test_source_default_and_original_callback_grammar(self):
  before=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True));after=ast.parse(Path('gpu_engine/transition.py').read_text());baseline=dump_default_program(before)
  self.assertEqual(dump_default_program(after),baseline);assert_hp_rng_mutations_detected(self,after,baseline)
  with self.assertRaises(ValueError):hit.check_compiled(hit.Compiled(self.helper.device))
  with self.assertRaises(ValueError):jax.jit(lambda:hit.check_compiled(self.helper).binding)()
  script=r'''
const root=process.argv[1],{Dex}=require(root+'/dist/sim/dex.js'),{Battle}=require(root+'/dist/sim/battle.js');const d=Dex.mod('gen3'),out=[];
for(const id of ['waterabsorb','voltabsorb','flashfire'])for(const hp of [50,100]){
 const effect=d.abilities.get(id),lines=[];const source={toString:()=> 'p1a: Actor'};
 const target={hp,maxhp:100,baseMaxhp:100,isActive:true,fainted:false,status:'',volatiles:{},toString:()=> 'p2a: Target',hasType:()=>false,
 getHealth(){return this.hp+'/100'},heal(n){const old=this.hp;this.hp=Math.min(100,this.hp+n);return this.hp-old}};
 const battle={effect,effectState:{},event:{target,source,effect:null},trunc:Math.trunc,runEvent:(id,t,s,e,n)=>n,heal:Battle.prototype.heal,
 add:(...xs)=>lines.push(xs.map(x=>typeof x==='function'?x.call(target):String(x)).join('|'))};
 target.addVolatile=()=>{if(hp===100)return false;effect.condition.onStart.call(battle,target);return true};
 const move=d.moves.get(id==='waterabsorb'?'hydropump':id==='voltabsorb'?'thunderbolt':'flamethrower');
 const result=effect.onTryHit.call(battle,target,source,move);out.push({id,hp,lines,result});}
process.stdout.write(JSON.stringify(out));'''
  values=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True))
  for value in values:
   abi={'waterabsorb':'Water Absorb','voltabsorb':'Volt Absorb','flashfire':'Flash Fire'}[value['id']]
   if value['hp']==100:line='-immune|p2a: Target|[from] ability: '+abi
   elif value['id']=='flashfire':line='-start|p2a: Target|ability: Flash Fire'
   else:line='-heal|p2a: Target|75/100|[from] ability: '+abi+'|[of] p1a: Actor'
   self.assertIsNone(value['result']);self.assertEqual(value['lines'],[line])
  self.assertTrue(bool(self.runtime['public'].device.no_damage[1:,0].all()))
 def component(self,b=8,*,reason=nd.MISS,mask=None):
  rows=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);no=j.zeros(b,bool);mask=yes if mask is None else mask
  cursor=life.register(rows+16001,**self.kw);physical=z+3;target=z+4
  mapping=j.full((b,2,6),-1,j.int32).at[:,0,3].set(0).at[:,1,4].set(0)
  cursor=cursor._replace(generation=j.ones((b,2),j.int32),turn=z+1,initialized=yes,physical_active=j.stack((physical,target),axis=1),public_slot=j.zeros((b,2),j.int32),encounter=j.ones((b,2),j.int32),established=j.ones((b,2),bool),physical_to_public=mapping,next_slot=j.ones((b,2),j.int32))
  tape=life._tape(cursor,self.producer,64);account=self.runtime['execution_compiled'];r=self.runtime['kernel_risk_compiled']
  tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes)
  carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,r.device),tape,r.device,active=yes);tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,r.device)))
  name='earthquake'if reason==nd.TYPE_IMMUNE else 'thunderbolt'if reason in (nd.VOLT_HEAL,nd.VOLT_FULL)else 'flamethrower'if reason in (nd.FLASH_START,nd.FLASH_REPEAT)else 'hydropump';move=z+self.metadata['vocabulary']['moves'].index(name)
  tape=public.move(tape,z,physical,target,move,eligible=mask);tape,ticket=ex.open_attempt(tape,ex.Site.DIRECT_HIT,z+1,active=mask)
  water=reason in (nd.WATER_HEAL,nd.WATER_FULL);volt=reason in (nd.VOLT_HEAL,nd.VOLT_FULL);flash=reason in (nd.FLASH_START,nd.FLASH_REPEAT);absorbed=water or volt or flash
  abi=self.metadata['vocabulary']['abilities'].index('waterabsorb'if water else 'voltabsorb'if volt else 'flashfire'if flash else 'pressure')
  typ=self.metadata['vocabulary']['types'].index('water'if water else 'electric'if volt else 'fire'if flash else 'ground'if reason==nd.TYPE_IMMUNE else 'water')
  old=z+(100 if reason in (nd.WATER_FULL,nd.VOLT_FULL)else 50);new=old+(25 if reason in (nd.WATER_HEAL,nd.VOLT_HEAL)else 0)
  oldvol=j.full(b,absorption.FLASH_FIRE if reason==nd.FLASH_REPEAT else 0,j.uint32);newvol=oldvol|j.uint32(absorption.FLASH_FIRE)if flash else oldvol
  accuracy=no if reason==nd.MISS else yes;hitflag=yes if reason==nd.TYPE_IMMUNE else no;immune=yes if reason==nd.TYPE_IMMUNE else no;num=z if reason==nd.TYPE_IMMUNE else z+1
  before=nd.Before(accuracy,z+typ,z+abi,z,old,z+100,oldvol,z,z+100,no,z,z+70)
  p=nd.prepare(tape,ticket,z,physical,target,move,before,self.runtime['direct_hit_context'],requested=mask)
  actual_absorb=absorption.try_hit(self.table.absorption,before.ability,move,before.move_type,before.status,
      j.full((b,2),self.metadata['vocabulary']['types'].index('normal'),j.int32),before.hp,before.maximum,before.sub_hp,before.volatile,eligible=mask&accuracy)
  p=nd.observed_absorption(p,actual_absorb)
  self.assertEqual(p.absorbed.hp.tolist(),new.tolist());self.assertEqual(p.absorbed.volatile.tolist(),newvol.tolist())
  w=hit.HitWrite(z+100,no,hitflag,immune,num,z+1,no,no,no,new,new,z+100)
  o=risk.Outcome(w.power,w.critical,w.hit,w.immune,w.numerator,w.denominator,p.absorbed.blocked,no,no,no,z,no)
  return tape,p,w,o,z+100,newvol,z,before.actor_hp_type,before.actor_hp_power
 def test_literal_branch_matrices_and_current_source_snapshot(self):
  for reason in range(1,9):
   with self.subTest(reason=reason):
    tape,p,w,o,hp,vol,sub,hpt,hpp=self.component(reason=reason);out=nd.finish(tape,p,w,o,hp,vol,sub,hpt,hpp)
    self.assertEqual(out.count.tolist(),[3]*8);self.assertEqual(out.execution.ledger.stage[:,0].tolist(),[2]*8)
    self.assertEqual(out.words[:,1,6].tolist(),[reason]*8);self.assertTrue(bool(nd.captured_matches(out,nd.Captured(p,w,hp,vol,sub,hpt,hpp),o).all()))
    self.assertTrue(bool(nd.owned_pair(out,j.array(1,j.int32),self.helper.device.no_damage)[0].all()))
    observed=risk.outcome_on_tape(out,o,eligible=j.ones(8,bool),active=j.ones(8,bool),captured_no_damage=nd.Captured(p,w,hp,vol,sub,hpt,hpp))
    self.assertEqual(observed.execution.risks.carry.risk.tolist(),[0]*8)
 def test_actual_miss_pure_type_and_named_absorbs_both_full_owners(self):
  arrays=self.f.base().arrays('pressure','pressure');b=5;pair=[]
  for row,(species,abi)in enumerate((('pidgeot','pressure'),('vaporeon','waterabsorb'),('houndoom','flashfire'),('jolteon','voltabsorb'),('pikachu','pressure'))):
   own_arrays={k:np.array(v[:1],copy=True)for k,v in arrays.items()};foe_arrays={k:np.array(v[1:2],copy=True)for k,v in arrays.items()}
   for a in (own_arrays,foe_arrays):a['moves'][:]=np.array([self.metadata['vocabulary']['moves'].index(n)for n in self.names]);a['pp'][:]=np.array([16,8,24,24]);a['item'][:]=0
   foe_arrays['species'][0,4]=self.metadata['vocabulary']['species'].index(species);foe_arrays['ability'][0,4]=self.metadata['vocabulary']['abilities'].index(abi)
   species_info=self.dex.species_info(species);foe_arrays['types'][0,4]=[self.metadata['vocabulary']['types'].index(t.lower())for t in species_info.types]+[0]*(2-len(species_info.types))
   pair.extend((own_arrays,foe_arrays))
  stacked={k:np.concatenate([a[k]for a in pair],axis=0)for k in arrays}
  with TemporaryDirectory()as tmp:
   p=Path(tmp)/'literal.npz';np.savez_compressed(p,**dict(stacked,metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8)));pool=source_provenance.load_constructor_pool(p)
   session=driver.construct(pool,np.arange(b*2).reshape(b,2),prepared=self.prepared,stream_ids=j.arange(17001,17001+b,dtype=j.int32),seed=31)
  self.assertEqual(session.state.core.admitted.tolist(),[True]*b);before=session.state.core.engine
  draws=j.zeros((b,2,12),j.int32).at[:,:,4].set(1).at[4,0,6].set(99)
  chance=tr.ChanceInputs(draws,*[j.zeros((b,2),j.int32)for _ in range(4)],j.zeros(b,j.int32));actions=j.array([[0,7],[1,7],[2,7],[3,7],[1,7]],j.int32)
  frames=[];original=self.runtime['frame']
  def capture(*args,**kwargs):result=original(*args,**kwargs);frames.append(result);return result
  self.runtime['frame']=capture
  try:result=driver.advance(session,actions,chance_inputs=chance)
  finally:self.runtime['frame']=original
  # Optional compact synthetic evidence permits writer-only diagnostics without
  # executing this constructor/kernel fixture again. It is never live data.
  capture=os.environ.get('POKEZERO_NODAMAGE_TEST_CAPTURE')
  if capture:
   report=result.state.core.frame_input_report;tape=frames[0].callbacks
   payload={k:np.asarray(getattr(before,k))for k in ('active','species','level','gender','hp','max_hp')}
   payload.update(words=np.asarray(tape.words),count=np.asarray(tape.count),switch_details=np.asarray(tape.switch_details),event_turn=np.asarray(tape.event_turn),
       owner_full_words=np.asarray(report.owner_full_words),owner_full_valid=np.asarray(report.owner_full_valid),owner_full_more=np.asarray(report.owner_full_more))
   np.savez_compressed(capture,**payload);self.assertLess(Path(capture).stat().st_size,100*1024)
  baseline=jax.jit(lambda state,action,draws:tr.step_with_chance(state,action,self.table,draws).state)(before,actions,chance)
  self.same(result.state.core.engine,baseline)
  self.assertEqual(result.state.core.admitted.tolist(),[True]*b,json.dumps(dict(risk=result.state.core.risks.risk.tolist(),refusal=frames[0].callbacks.refusal.tolist(),ledger=result.state.core.execution.refusal.tolist(),witness=result.state.core.frame_input_report.refusal.tolist(),owner=[o.reason.tolist()for o in result.owners])))
  raw=frames[0].callbacks.words;at=j.argmax(raw[:,:,0]==33,axis=1);self.assertEqual(raw[j.arange(b),at,6].tolist(),[nd.TYPE_IMMUNE,nd.WATER_FULL,nd.FLASH_START,nd.VOLT_FULL,nd.MISS])
  ledger=result.state.core.execution;active=(ledger.site==10)&(ledger.serial>0);self.assertTrue(bool(j.all(j.where(active,ledger.stage==3,True))))
  report=result.state.core.frame_input_report
  for owner in (0,1):
   for row in range(b):
    actual=np.asarray(report.owner_full_words[row,owner])[np.asarray(report.owner_full_valid[row,owner])]
    np.testing.assert_array_equal(actual,self.reference_words(frames[0].callbacks,row,owner,before))
  for owner in result.owners:self.assertEqual(owner.full_line_consumers_complete.tolist(),[True]*b);self.assertFalse(bool(owner.B4_qualified.any()))

 def reference_words(self,tape,row,owner,before):
  # An independent original writer expansion of this literal fixture's raw
  # calls, including all empty preludes and owner-only Pressure lines.
  def species_name(engine):return self.catalog.species.names[int(self.runtime['public'].device.causal.species[int(engine)])]
  names={side:species_name(before.species[row,side,int(before.active[row,side])])for side in (0,1)}
  ident={side:f'p{side+1}a: '+names[side]for side in (0,1)}
  def switch(side,sp,level,gender,hp,maximum):
   names[side]=species_name(sp);ident[side]=f'p{side+1}a: '+names[side]
   if side!=owner:hp=100 if hp==maximum else min(99,(100*hp+maximum-1)//maximum);maximum=100
   suffix={0:'',1:', M',2:', F'}[int(gender)]
   return f'|switch|{ident[side]}|{names[side]}, L{int(level)}{suffix}|{int(hp)}/{int(maximum)}'
  prefix=[]
  for side in (0,1):
   slot=int(before.active[row,side]);prefix.append(switch(side,*[int(getattr(before,k)[row,side,slot])for k in ('species','level','gender','hp','max_hp')]))
  prefix.append(f'|-ability|{ident[owner]}|Pressure|[silent]');prefix.append('|turn|1')
  wire,_=public_wire.project_lines(public_wire.WireState(),prefix,**v4.wire_kwargs(self.catalog));lines=[]
  for index,w in enumerate(np.asarray(tape.words[row,:int(tape.count[row])])):
   kind=int(w[0]);side=int(w[3])
   if kind==life.SWITCH_INSTALLED:
    sp,level,gender,hp,maximum,status=map(int,tape.switch_details[row,index]);self.assertEqual(status,0)
    lines.append(switch(side,sp,level,gender,hp,maximum))
   elif kind==24:
    if side==owner:lines.append(f'|-ability|{ident[side]}|Pressure|[silent]')
   elif kind==life.TURNLOOP_PREFIX:
    self.assertEqual(w.tolist(),[35,-1,-1,-1,-1,0,0,0,0,0])
    lines.extend(['|','|t:|1'])
   elif kind==public.MOVE:
    move=self.catalog.dex.move_info(self.metadata['vocabulary']['moves'][int(w[5])]).name
    lines.append(f'|move|{ident[int(w[1])]}|{move}|{ident[side]}')
   elif kind==nd.TARGET_NO_DAMAGE:
    reason=int(w[6]);target=ident[side]
    if reason==nd.MISS:lines.append(f'|-miss|{ident[int(w[1])]}|{target}')
    elif reason==nd.TYPE_IMMUNE:lines.append('|-immune|'+target)
    elif reason in (nd.WATER_FULL,nd.VOLT_FULL,nd.FLASH_REPEAT):
     ability={nd.WATER_FULL:'Water Absorb',nd.VOLT_FULL:'Volt Absorb',nd.FLASH_REPEAT:'Flash Fire'}[reason]
     lines.append(f'|-immune|{target}|[from] ability: {ability}')
    elif reason==nd.FLASH_START:lines.append(f'|-start|{target}|ability: Flash Fire')
    else:self.fail('unrepresented literal callback '+str(reason))
   elif kind==nd.ABSORB_FACTS:pass
   elif kind==life.ACTION_END:lines.append('|')
   elif kind==life.UPKEEP:lines.append('|upkeep')
   elif kind==life.TURN:lines.append(f'|turn|{int(tape.event_turn[row,index])}')
   else:self.fail('unrepresented literal raw '+str(kind))
  _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog))
  return np.asarray([e.words()for packet in packets for e in packet.events],np.int32)
