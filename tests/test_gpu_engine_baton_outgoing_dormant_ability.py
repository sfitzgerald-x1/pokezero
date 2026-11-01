"""Source-qualified dormant outgoing callbacks; no turn, model or FULL replay."""
import ast,hashlib,json,subprocess,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import jax.numpy as j
import numpy as np
import test_gpu_engine_baton_pass_transaction as fixture
from gpu_engine import engine_baton_pass_callbacks as cb,engine_execution_completeness as ex
from gpu_engine import engine_kernel_frame_risk as risk,engine_lifecycle_callbacks as life
from gpu_engine import engine_public_events as public,state
from _showdown_root import showdown_root
BASE='6a55ed80787129fdfab289f20f97193e263f60c2'

class DormantOutgoingTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  original=fixture.fixtures.SelfBoostInputTests.setUpClass.__func__
  def with_abilities(base):
   original(base)
   for name in ('insomnia','naturalcure'):
    if name not in base.metadata['vocabulary']['abilities']:base.metadata['vocabulary']['abilities'].append(name)
  with patch.object(fixture.fixtures.SelfBoostInputTests,'setUpClass',classmethod(with_abilities)):
   fixture.BatonTransactionTests.setUpClass.__func__(cls)
  cls.data=cls.helper.device.baton_pass

 def transaction(self):
  b=8;r=j.arange(b);z=j.zeros(b,j.int32);yes=j.ones(b,bool);side=r%2
  ids=[self.metadata['vocabulary']['abilities'].index(n)for n in ('insomnia','naturalcure')]
  s=state.empty_state(b)._replace(hp=j.full((b,2,6),100,j.int32),max_hp=j.full((b,2,6),100,j.int32),ability=j.full((b,2,6),self.ability,j.int32),base_ability=j.full((b,2,6),self.ability,j.int32))
  actor=j.asarray([ids[i%2]for i in range(b)],j.int32)
  s=s._replace(ability=s.ability.at[r,side,z].set(actor),base_ability=s.base_ability.at[r,side,z].set(actor))
  cursor=life.register(r+193001,**self.kw);t=life._tape(cursor,self.producer,64)
  for owner in (0,1):t=life._reveal(t,s,z+owner,z-1,z,eligible=yes,replacement=~yes,baton=~yes,initial=True)
  cursor=t.cursor._replace(initialized=yes)
  t=life._tape(cursor,self.producer,64);a=self.runtime['execution_compiled'];rc=self.runtime['kernel_risk_compiled']
  t=ex.begin_on_tape(t,ex.initialize(cursor.stream_id,compiled=a),a.device,active=yes)
  carry=risk.begin_frame_device(risk.initialize_device(t.stream_id,rc.device)._replace(frame=t.execution.ledger.frame),t,rc.device,active=yes)
  t=t._replace(execution=t.execution._replace(risks=risk.Context(carry,rc.device)))
  order=j.broadcast_to(j.arange(6),(b,2,6));h=cb.empty_history(s,t.cursor,order)
  # Declared literal before-public/request fixture. No FULL/maturation claim.
  proof=cb.PublicProof(j.zeros((b,2,7),j.int32),j.ones((b,2),bool),j.zeros((b,2),bool),j.zeros((b,2),bool))
  tr=cb.Transaction(h,proof,yes,z+3,order,cb.empty_record(h))
  t=t._replace(execution=t.execution._replace(baton_transaction=tr));t=life.begin(t,s,yes,~yes)
  t=public.move(t,side,z,z,z+self.move,eligible=yes)
  return t,s,side,z,yes

 def test_both_source_boundaries_current_none_only_and_recipient_closed(self):
  t,s,side,z,yes=self.transaction();context=self.runtime['direct_hit_context'];move=z+self.move
  self.assertEqual(cb._bound(t,self.data).tolist(),[True]*8)
  self.assertEqual(cb._quiet(s,side,z,self.data).tolist(),[False]*8)
  _,p=cb.begin_start(t,s,side,move,move,context,acts=yes,eligible=yes)
  self.assertEqual(p.valid.tolist(),[True]*8)
  for field,value in [('status',1),('status',4),('status',5),('status_counter',1),('base_ability',self.ability),('confusion_counter',1),('hp',0)]:
   bad=s._replace(**{field:getattr(s,field).at[0,side[0],0].set(value)})
   _,q=cb.begin_start(t,bad,side,move,move,context,acts=yes,eligible=yes)
   self.assertEqual(q.valid.tolist(),[False]+[True]*7,(field,value))
  for kwargs in (dict(acts=~yes,caller=move),dict(acts=yes,caller=move+1)):
   _,q=cb.begin_start(t,s,side,move,kwargs['caller'],context,acts=kwargs['acts'],eligible=yes)
   self.assertEqual(q.valid.tolist(),[False]*8)
  s=s._replace(baton_pass=s.baton_pass.at[j.arange(8),side].set(True))
  tr=t.execution.baton_transaction;h=tr.history._replace(side=side,private=cb.snapshot(s),boosts=s.boosts)
  t=t._replace(execution=t.execution._replace(baton_transaction=tr._replace(history=h)))
  _,q=cb.before_copy(t,s,side,z+3,context,eligible=yes);self.assertEqual(q.valid.tolist(),[True]*8)
  for field,value in [('status',1),('status_counter',1),('base_ability',self.ability),('hp',99)]:
   bad=s._replace(**{field:getattr(s,field).at[0,side[0],0].set(value)})
   _,q=cb.before_copy(t,bad,side,z+3,context,eligible=yes);self.assertEqual(q.valid.tolist(),[False]+[True]*7,(field,value))
  # The incoming ability still uses the old strict quiet table, even at NONE.
  bad=s._replace(ability=s.ability.at[0,side[0],3].set(s.ability[0,side[0],0]),base_ability=s.base_ability.at[0,side[0],3].set(s.ability[0,side[0],0]))
  tr=t.execution.baton_transaction;h=tr.history._replace(private=cb.snapshot(bad))
  t2=t._replace(execution=t.execution._replace(baton_transaction=tr._replace(history=h)))
  _,q=cb.before_copy(t2,bad,side,z+3,context,eligible=yes);self.assertEqual(q.valid.tolist(),[False]+[True]*7)
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   bad=t._replace(**{name:getattr(t,name).at[0].add(j.uint32(1))})
   self.assertEqual(cb._bound(bad,self.data).tolist(),[False]*8,name)
  tables=t.execution.risks.tables
  badtables=tables._replace(baton_outgoing_dormant_abilities=~tables.baton_outgoing_dormant_abilities)
  bad=t._replace(execution=t.execution._replace(risks=t.execution.risks._replace(tables=badtables)))
  self.assertEqual(cb._bound(bad,self.data).tolist(),[False]*8)
  legacy=self.data._replace(outgoing_dormant_abilities=None)
  self.assertEqual(cb._quiet(s,side,z,legacy,outgoing_dormant=True).tolist(),[False]*8)
  for value in (j.broadcast_to(self.data.outgoing_dormant_abilities,(8,len(self.data.quiet_abilities))),self.data.outgoing_dormant_abilities.astype(j.int32)):
   with self.assertRaises(ValueError):cb.shape(self.data._replace(outgoing_dormant_abilities=value))

 def test_original_callbacks_no_words_at_none_and_active_branches_stay_excluded(self):
  script=r"""
const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),H=x=>require('crypto').createHash('sha256').update(''+x).digest('hex'),rows=[];
for(const [id,hook] of [['insomnia','onUpdate'],['naturalcure','onSwitchOut']])for(const status of ['', 'slp','par']){
 const lines=[],p={status,hp:71,pp:[4,5,6,7],clearStatus(){this.status=''},cureStatus(){this.status=''},toString(){return 'p1a: Actor'}};
 const before={hp:p.hp,pp:[...p.pp]};d.abilities.get(id)[hook].call({add(...args){lines.push('|'+args.map(String).join('|'))}},p);
 rows.push({id,status,after:p.status,lines,hp:p.hp,pp:p.pp,hash:H(d.abilities.get(id)[hook]),before});
}
console.log(JSON.stringify(rows));
"""
  rows=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True))
  for row in rows:
   self.assertEqual(row['hp'],row['before']['hp']);self.assertEqual(row['pp'],row['before']['pp'])
   self.assertEqual(row['hash'],cb.INSOMNIA_UPDATE_SHA256 if row['id']=='insomnia'else cb.NATURAL_CURE_SWITCH_OUT_SHA256)
   if not row['status']:self.assertEqual(row['lines'],[]);self.assertEqual(row['after'],'')
  self.assertEqual(rows[1]['lines'],['|-activate|p1a: Actor|ability: Insomnia'])
  self.assertEqual(rows[5]['lines'],['|-curestatus|p1a: Actor|par|[from] ability: Natural Cure'])
  for name in ('insomnia','naturalcure'):
   index=self.metadata['vocabulary']['abilities'].index(name)
   self.assertFalse(bool(self.data.quiet_abilities[index]));self.assertTrue(bool(self.data.outgoing_dormant_abilities[index]))
  # An altered effective original handler fails before capability publication.
  with patch.object(cb,'INSOMNIA_UPDATE_SHA256','0'*64):
   with self.assertRaisesRegex(ValueError,'dormant outgoing'):cb.compile_data(self.producer,self.metadata,self.rules)

 def test_source_defaults_consumers_and_call_scope_unchanged(self):
  for name in ('transition.py','entry.py','baton.py','state.py','mechanics.py','engine_retained_encoder_driver.py','engine_full_v4_event_device.py','engine_lifecycle_event_device.py'):
   self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show',('7e9580207defb397e81313d0ace17d38eb8a6aea'if name=='engine_retained_encoder_driver.py'else BASE)+':gpu_engine/'+name]),name)
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_baton_pass_callbacks.py'],text=True));new=ast.parse(Path(cb.__file__).read_text())
  changed={'compile_data','shape','_bound','_data','_ability_quiet','_quiet','_start_diagnostics','begin_start','before_copy'}
  fns=lambda t:{n.name:ast.dump(n,include_attributes=False)for n in t.body if isinstance(n,ast.FunctionDef)and n.name not in changed}
  self.assertEqual(fns(old),fns(new))
  actual=[]
  for fn in (n for n in new.body if isinstance(n,ast.FunctionDef)):
   for call in (n for n in ast.walk(fn)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='_quiet'):
    if any(k.arg=='outgoing_dormant'and isinstance(k.value,ast.Constant)and k.value.value is True for k in call.keywords):actual.append((fn.name,ast.unparse(call.args[2])))
  self.assertEqual(actual,[('begin_start','physical'),('before_copy','old')])
  from gpu_engine import engine_source_frame_witness as witness
  witness._source()
  for names in ('gpu_engine.benchmark_retained_games,gpu_engine.engine_baton_pass_callbacks,gpu_engine.engine_full_v4_event_device','gpu_engine.engine_full_v4_event_device,gpu_engine.engine_baton_pass_callbacks,gpu_engine.benchmark_retained_games'):
   subprocess.run([sys.executable,'-B','-c','import importlib;'+''.join('importlib.import_module('+repr(n)+');'for n in names.split(','))],check=True,capture_output=True,text=True,timeout=30)

if __name__=='__main__':unittest.main()
