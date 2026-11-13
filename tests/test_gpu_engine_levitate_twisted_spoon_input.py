"""Dormant defending Twisted Spoon, exact existing Levitate route only."""
import ast,json,subprocess,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_no_damage_callbacks as nd,engine_direct_hit_controls as hit
from gpu_engine import engine_public_events as events,engine_request_source as request,engine_own_facts as own,engine_leftovers_callbacks as leftovers
from gpu_engine import engine_retained_encoder_driver as driver,engine_execution_completeness as ex,engine_lifecycle_callbacks as life,engine_kernel_frame_risk as risk
from gpu_engine import engine_full_v4_event_device as full,engine_source_frame_witness as witness,transition,public_wire,public_inputs as inputs,v4_encoder as v4
from pokezero.dex import normalize_id
from _showdown_root import showdown_root
import test_gpu_engine_earthquake_levitate_input as fixture
BASE='e481228626c9cfdfe2a2989871672392aa3e5859'
class TwistedSpoonLevitateTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.EarthquakeLevitateTests.setUpClass();f=fixture.EarthquakeLevitateTests;cls.f=f
  for name in ('dex','catalog','levitate','move'):setattr(cls,name,getattr(f,name))
  cls.metadata=deepcopy(f.metadata)
  # Exact existing generate_teams.cjs namespace rule, not an invented item ID.
  script="const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3');console.log(JSON.stringify(['',...Array.from(new Set(d.items.all().filter(x=>x.exists&&x.gen<=3).map(x=>x.id))).sort()]));"
  cls.metadata['vocabulary']['items']=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));cls.item=cls.metadata['vocabulary']['items'].index('twistedspoon')
  cls.table=transition.tables_from_metadata(cls.metadata);cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,exposure=events.HealthExposure(events.SHOWDOWN,'gen3randombattle',True,False));cls.rules=request.load_source_rules(showdown_root(),cls.metadata)
  req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer);facts=own.compile_own_facts(req,cls.catalog)
  cls.helper=hit.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules);cls.legacy=nd.compile_data(cls.producer)
  cls.prepared=driver.prepare(req,facts,public_mode='retained-full',account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)));cls.runtime=driver._checked(cls.prepared)
 def component(self,b=8,mask=None):
  t,p,w,o,tail=fixture.EarthquakeLevitateTests.component(self,b=b,mask=mask);return t,p._replace(before=p.before._replace(item=j.full(b,self.item,j.int32))),w,o,tail
 def finish(self,parts):
  t,p,w,o,tail=parts;return nd.finish(t,p,w,o,*tail,target_types=p.before.target_types,item=p.before.item,ability=p.before.ability)
 def test_exact_pinned_dormant_source_and_mask_provenance(self):
  self.assertEqual((len(self.metadata['vocabulary']['items']),self.item),(130,122));d=self.helper.device.no_damage;lower=self.runtime['public'].device.causal.no_damage
  self.assertTrue(bool(d.levitate_items[self.item]));self.assertTrue(bool(lower.levitate_items[self.item]));self.assertFalse(bool(self.legacy.levitate_items[self.item]))
  self.assertFalse(bool(lower.levitate_source));self.assertTrue(bool(d.levitate_source))
  self.assertEqual(np.flatnonzero(np.asarray(d.levitate_items)).tolist(),[0,self.metadata['vocabulary']['items'].index('leftovers'),self.item])
  # Independent lower copy transfers ONLY the original checked item mask;
  # legacy optional source flags and every other data field remain unchanged.
  for name in nd.Data._fields:
   if name not in ('binding','levitate_items'):np.testing.assert_array_equal(getattr(lower,name),getattr(self.legacy,name))
  with self.assertRaises(ValueError):nd.compile_data(self.producer,levitate_item_compiled=self.f.helper)
  with self.assertRaises(ValueError):nd.compile_data(self.producer,levitate_item_compiled=d)
  # Pin every actual on* property, including nonfunction priorities. A changed
  # function or an additional reactive hook cannot be treated as generic quiet.
  run=nd.subprocess.run
  def changed(*args,**kw):
   r=run(*args,**kw);data=json.loads(r.stdout);data['twistedspoon']['hooks'].append(['onDamagingHit','function','a'*64]);return subprocess.CompletedProcess(r.args,r.returncode,json.dumps(data),r.stderr)
  with patch.object(nd.subprocess,'run',changed):
   with self.assertRaisesRegex(ValueError,'Twisted Spoon dormant source'):nd.compile_data(self.producer,self.metadata,self.rules)
  script=r'''const root=process.argv[1],d=require(root+'/dist/sim/dex.js').Dex.mod('gen3'),P=require(root+'/dist/sim/pokemon.js').Pokemon,logs=[],events=[];const battle={gen:3,field:{pseudoWeather:{}},runEvent:id=>(events.push(id),true),suppressingAbility:()=>false,dex:d,add:(...x)=>logs.push(x.map(v=>typeof v==='object'?'p2a: Target':v).join('|'))},target={battle,volatiles:{},item:'twistedspoon',ignoringItem:()=>false,hasType:t=>t==='Normal',hasAbility:a=>a==='levitate',isGrounded:P.prototype.isGrounded};const result=P.prototype.runImmunity.call(target,d.moves.get('earthquake'),true);console.log(JSON.stringify({result,logs,events}));'''
  original=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(original,dict(result=False,logs=['-immune|p2a: Target|[from] ability: Levitate'],events=['NegateImmunity']))
 def test_actual_producer_to_lower_owned_mask_and_current_guards(self):
  parts=self.component(mask=j.array([True]*6+[False]*2));t,p,w,o,tail=parts;out=self.finish(parts);cap=nd.Captured(p,w,*tail,p.before.target_types,p.before.item,p.before.ability)
  self.assertEqual(nd.scheduled(p,w).tolist(),[True]*6+[False]*2);self.assertEqual(nd.captured_matches(out,cap,o).tolist(),[True]*6+[False]*2)
  lower=self.runtime['public'].device.causal.no_damage;header,facts=nd.words(p,w)
  self.assertEqual(nd.valid_raw_pair(header,facts,lower).tolist(),[True]*6+[False]*2);self.assertFalse(bool(nd.valid_raw_pair(header,facts,self.legacy).any()))
  self.assertEqual(nd.owned_pair(out,1,lower)[0].tolist(),[True]*6+[False]*2)
  for name,value in [('item',j.full(8,self.metadata['vocabulary']['items'].index('magnet'),j.int32)),('volatile',j.ones(8,j.uint32)),('target_types',p.before.target_types.at[:,0].set(int(lower.flying_type))),('item',None)]:self.assertFalse(bool(nd.scheduled(p._replace(before=p.before._replace(**{name:value})),w).any()),name)
  self.assertFalse(bool(nd.scheduled(p._replace(caller_move=p.move+1),w).any()))
  for field,value in [('item',cap.item.at[0].set(0)),('ability',cap.ability.at[0].set(0)),('target_types',cap.target_types.at[0,0].set(self.metadata['vocabulary']['types'].index('water')))]:self.assertEqual(nd.captured_matches(out,cap._replace(**{field:value}),o).tolist(),[False]+[True]*5+[False]*2)
  sticky=out._replace(refusal=out.refusal|256);got=risk.outcome_on_tape(sticky,o,eligible=p.requested,active=p.requested,captured_no_damage=cap);self.assertTrue(bool(j.all((got.execution.risks.carry.risk[:6]&risk.UPSTREAM)!=0)))
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_no_damage_callbacks.py'],text=True));new=ast.parse(Path(nd.__file__).read_text())
  self.assertEqual(ast.dump(ast.Module(body=[n for n in old.body if not(isinstance(n,ast.FunctionDef)and n.name=='compile_data')],type_ignores=[])),ast.dump(ast.Module(body=[n for n in new.body if not(isinstance(n,ast.FunctionDef)and n.name=='compile_data')],type_ignores=[])))
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
  # state.py and items.py are byte-certified against the reviewed SOURCE_PINS by transition_observer_wiring.
  for name in ('mechanics.py','absorption.py','engine_kernel_frame_risk.py','engine_execution_completeness.py'):
   self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+name]))
  witness._source()
 def test_one_existing_full_grammar_packet_no_private_revelation(self):
  # One owner, two mirrored rows. Reuse existing Levitate grammar, not a FULL
  # matrix or a new lexical permission. The mask must reach the ACTUAL lower.
  t,p,w,o,tail=self.component(b=2);t=self.finish((t,p,w,o,tail));lower=self.runtime['public'].device.causal.no_damage
  self.assertEqual(nd.owned_pair(t,1,lower)[0].tolist(),[True,True]);self.assertFalse(bool(lower.levitate_source));self.assertTrue(bool(self.runtime['kernel_risk_compiled'].device.levitate_immunity))
  kw=v4.wire_kwargs(self.catalog);ns={k:np.asarray(getattr(self.runtime['public'].device.namespaces,k))for k in full._NAMES};initial=[];expected=[];engine_ids=self.metadata['vocabulary']['species'];species_names=list(self.catalog.species.names[1:4])
  for row in range(2):
   side=row%2;name=species_names[2];ids=[f'p{i+1}a: '+name for i in(0,1)];lines=[]
   for previous in species_names[:2]:
    for s in(0,1):lines.append(f'|switch|p{s+1}a: {previous}|{previous}, L80, M|100/100')
   for s in(0,1):lines.append(f'|switch|{ids[s]}|{name}, L80, M|100/100')
   lines.append('|turn|1');wire,packets=public_wire.project_lines(public_wire.WireState(),lines,**kw);initial.append([e for packet in packets for e in packet.events]);_,packets=public_wire.project_lines(wire,[f'|move|{ids[side]}|Earthquake|{ids[1-side]}',f'|-immune|{ids[1-side]}|[from] ability: Levitate'],**kw);expected.append(np.asarray([e.words()for packet in packets for e in packet.events],np.int32))
  pub=v4.initialize(self.catalog,stream_ids=t.stream_id,owner_sides=j.zeros(2,j.int32))
  for chunk in inputs.pack_chunks(np.asarray(t.stream_id),initial,**ns):pub,error=v4.fold_chunk(pub,chunk,self.catalog);self.assertFalse(bool(error.any()))
  species=j.array([[engine_ids.index(normalize_id(self.catalog.species.names[int(pub.mons.species[r,s,2])]))for s in(0,1)]for r in range(2)],j.int32);cursor=t.cursor._replace(revealed_species_id=species);t=t._replace(cursor=cursor)
  stream=full.register(t.stream_id,j.zeros(2,j.int32),compiled=self.runtime['public']);stream=stream._replace(causal=stream.causal._replace(cursor=cursor,sequence=j.full(2,7,j.int32)),sequence=pub.cursor.sequence,live=j.ones((2,2),bool),species=pub.mons.species,status=pub.mons.status,hp=pub.mons.hp_numerator,maximum=pub.mons.hp_denominator)
  frame=life.LifecycleResult(None,cursor,t,life.snapshot(cursor),life.snapshot(cursor),j.zeros(2,bool),j.zeros(2,j.int32))
  consume=jax.jit(lambda ledger,stream,frame,pub:ex.consume_frame_device(ledger,frame,stream,pub,active=j.ones(2,bool),tables=self.runtime['execution_compiled'].device,adapter_tables=self.runtime['public'].device,catalog=self.catalog));ledger,got=consume(t.execution.ledger,stream,frame,pub)
  self.assertEqual(got.projection.error.tolist(),[0,0]);self.assertEqual(ledger.stage[:,0].tolist(),[3,3]);self.assertEqual(ledger.refusal.tolist(),[0,0]);captures=[]
  for row in range(2):
   actual=np.asarray(got.projection.frames.words).transpose(1,0,2,3)[row][np.asarray(got.projection.frames.valid).transpose(1,0,2)[row]];np.testing.assert_array_equal(actual,expected[row]);captures.append(dict(owner=0,row=row,actual=actual.tolist(),expected=expected[row].tolist(),private_item=int(p.before.item[row])))
  for name in('types','items'):
   for before,after in zip(jax.tree.leaves(getattr(pub.context,name)),jax.tree.leaves(getattr(got.public.context,name)),strict=True):np.testing.assert_array_equal(before,after)
  Path('/tmp/pokezero-levitate-twisted-spoon-original-full-compact.json').write_text(json.dumps(captures,sort_keys=True)+'\n')
