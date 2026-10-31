"""NON_GAME status schema/typed-write components, not eligible game turns."""
import json,unittest
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
import test_gpu_engine_self_heal_owning as fixtures
from gpu_engine import engine_self_heal_callbacks as cb,engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_lifecycle_callbacks as life,engine_full_v4_event_device as full
from gpu_engine import engine_public_events as public,public_wire,v4_encoder as v4,state
TOKENS=('','slp','par','brn','frz','psn','tox') # Original PUBLIC status ordering.
MAP=(0,2,3,5,6,1,4) # Original mechanics->PUBLIC status ordering.

class SelfHealStatusSchemasTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.SelfHealOwningTests.setUpClass();cls.f=fixtures.SelfHealOwningTests
 def test_original_schema_words_all_seven_public_statuses(self):
  f=self.f;name=f.catalog.species.names[1];ids=('p1a: '+name,'p2a: '+name)
  prior,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ids[s]}|{name}, L80, M|100/100'for s in(0,1)]+['|turn|1'],**v4.wire_kwargs(f.catalog));d=f.runtime['public'].device;values=[]
  for status,token in enumerate(TOKENS):
   _,pack=public_wire.project_line(prior,f'|-heal|{ids[0]}|71/100'+(' '+token if token else''),**v4.wire_kwargs(f.catalog));want=np.asarray([e.words()for e in pack.events],np.int32);key=int(d.self_heal[status]);actual=np.asarray(d.templates[key])[:int(d.length[key])];values.append(dict(public_status=status,token=token,actual=actual.tolist(),original=want.tolist()))
  # Persist all literal schemas BEFORE assertions, preserving a failed-before
  # counterexample without another expensive component rerun.
  path=Path('/tmp/pokezero-self-heal-status-schema-current.json');path.write_text(json.dumps(values,sort_keys=True)+'\n')
  for v in values:
   with self.subTest(status=v['public_status'],token=v['token']):np.testing.assert_array_equal(v['actual'],v['original'])
 def test_all_seven_statuses_both_owners_exact_words_and_saved_burn_toxic_operands(self):
  f=self.f;b=8;rows=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool)
  side=j.array([0,1,1,1,0,1,0,1],j.int32);active=j.stack((z+3,z+4),axis=1).at[2,1].set(0).at[4,0].set(0);physical=active[rows,side];at=(rows,side,physical)
  statuses=j.array([0,1,2,3,4,5,6,4],j.int32)
  old=j.array([77,77,128,77,79,77,77,77],j.int32);maximum=j.array([239,239,262,239,215,239,239,239],j.int32);new=j.minimum(maximum,old+maximum//2)
  s=state.empty_state(b)._replace(active=active,gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),species=j.broadcast_to(j.array([1,2])[None,:,None],(b,2,6)),level=j.full((b,2,6),80,j.int32))
  s=s._replace(hp=s.hp.at[at].set(old),max_hp=s.max_hp.at[at].set(maximum),status=s.status.at[at].set(statuses))
  kw=dict(catalog=f.producer,expected_binding=f.producer.binding,expected_engine_binding=np.asarray(f.producer.engine_binding,np.uint32));start=life.register(rows+83001,**kw);initial=life._tape(start,f.producer,64)
  for seat in(0,1):initial=life._reveal(initial,s,z+seat,z-1,active[:,seat],eligible=yes,replacement=~yes,baton=~yes,initial=True)
  initial=initial._replace(cursor=initial.cursor._replace(initialized=yes));initial=life._turn(initial,yes);frame0=life.LifecycleResult(s,initial.cursor,initial,life.snapshot(start),life.snapshot(initial.cursor),~yes,z);consumers=[]
  for owner in(0,1):
   stream=full.register(start.stream_id,z+owner,compiled=f.runtime['public']);pub=v4.initialize(f.catalog,stream_ids=start.stream_id,owner_sides=z+owner);got=f.fold(stream,frame0,pub);self.assertEqual(got.projection.error.tolist(),[0]*b);consumers.append((got.projection.state,got.public))
  t=life._tape(initial.cursor,f.producer,64);account=f.runtime['execution_compiled'];rt=f.runtime['kernel_risk_compiled'].device;t=ex.begin_on_tape(t,ex.initialize(start.stream_id,compiled=account),account.device,active=yes);carry=risk.begin_frame_device(risk.initialize_device(t.stream_id,rt),t,rt,active=yes);t=t._replace(execution=t.execution._replace(risks=risk.Context(carry,rt)))
  names=['recover']*8;names[2]='synthesis';moves=j.array([f.metadata['vocabulary']['moves'].index(n)for n in names],j.int32)
  t=public.move(t,side,physical,active[rows,1-side],moves,eligible=yes);t,p=cb.begin(t,s,side,moves,f.runtime['direct_hit_context'],f.table,eligible=yes)
  after=s._replace(hp=s.hp.at[at].set(new));t,c=cb.finish(t,p,after,z,yes);self.assertTrue(bool(cb.captured_matches(t,c).all()))
  # Literal measured RAW45 payloads, but new authored NON_GAME addresses. No
  # R29 root construction/action/physics/encoding replay or empirical claim.
  self.assertEqual(t.words[2,1,5:9].tolist(),[128,259,262,2]);self.assertEqual(t.words[4,1,5:9].tolist(),[79,186,215,4])
  frame=life.LifecycleResult(after,t.cursor,t,life.snapshot(initial.cursor),life.snapshot(t.cursor),~yes,z);all_actual=[]
  for owner,(stream,pub)in enumerate(consumers):
   got=f.fold(stream,frame,pub);all_actual.append(dict(owner=owner,error=got.projection.error.tolist(),words=got.projection.frames.words.tolist(),valid=got.projection.frames.valid.tolist()))
   Path('/tmp/pokezero-self-heal-status-full-current.json').write_text(json.dumps(all_actual,sort_keys=True)+'\n')
   self.assertEqual(got.projection.error.tolist(),[0]*b)
   for row in range(b):
    ids={seat:f'p{seat+1}a: '+f.catalog.species.names[seat+1]for seat in(0,1)};actor=int(side[row]);token=TOKENS[MAP[int(statuses[row])]]
    def health(hp,mx,seat):
     if owner==seat:return f'{hp}/{mx}'
     pct=(100*hp+mx-1)//mx
     return f'{99 if hp<mx and pct==100 else pct}/100'
    lines=[]
    for seat in(0,1):
     pos=int(s.active[row,seat]);status=int(s.status[row,seat,pos]);tail=TOKENS[MAP[status]];lines.append(f'|switch|{ids[seat]}|{f.catalog.species.names[seat+1]}, L80, M|{health(int(s.hp[row,seat,pos]),int(s.max_hp[row,seat,pos]),seat)}'+(' '+tail if tail else''))
    wire,_=public_wire.project_lines(public_wire.WireState(),lines+['|turn|1'],**v4.wire_kwargs(f.catalog));_,packets=public_wire.project_lines(wire,[f'|move|{ids[actor]}|{f.dex.move_info(names[row]).name}|{ids[actor]}',f'|-heal|{ids[actor]}|{health(int(new[row]),int(maximum[row]),actor)}'+(' '+token if token else'')],**v4.wire_kwargs(f.catalog));want=np.asarray([e.words()for p in packets for e in p.events],np.int32);actual=np.asarray(got.projection.frames.words[:,row])[np.asarray(got.projection.frames.valid[:,row])];np.testing.assert_array_equal(actual,want)
  for key in ('status','pp','keys','moves','boosts','ability','item'):np.testing.assert_array_equal(getattr(s,key),getattr(after,key))

if __name__=='__main__':unittest.main()
