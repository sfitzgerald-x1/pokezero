"""Source/current-boundary components only; no turn, FULL replay or model."""
import ast,hashlib,json,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
import test_gpu_engine_baton_outgoing_dormant_ability as original
from gpu_engine import engine_baton_pass_callbacks as cb,engine_kernel_frame_risk as risk
from gpu_engine import engine_outcome_diagnostics as diag,public_active
from _showdown_root import showdown_root

BASE='fd3256d2305a0770764e737640fc80157156cb29'

class InsomniaBurnPoisonTests(unittest.TestCase):
 setUpClass=classmethod(original.DormantOutgoingTests.setUpClass.__func__)
 transaction=original.DormantOutgoingTests.transaction

 def public_roots(self,t):
  cursor=t.cursor;b=len(t.count)
  def view():return NS(cursor=NS(stream_id=cursor.stream_id,turn=cursor.turn,phase=cursor.phase),mons=NS(active=cursor.public_slot,boosts=j.zeros((b,2,7),j.int32)),context=NS(active=NS(encounter=cursor.encounter,volatiles=j.zeros((b,2,len(public_active.NAMES)),bool)),tendency=NS(baton=j.zeros((b,2),bool))))
  return view(),view()

 def test_current_both_boundary_public_join_and_status_conservation(self):
  t,s,side,z,yes=self.transaction();r=j.arange(8);move=z+self.move;context=self.runtime['direct_hit_context'];insomnia=self.metadata['vocabulary']['abilities'].index('insomnia');natural=self.metadata['vocabulary']['abilities'].index('naturalcure')
  statuses=j.array([2,3,3,2,2,3,3,2],j.int32)
  s=s._replace(status=s.status.at[r,side,z].set(statuses),ability=s.ability.at[r,side,z].set(insomnia),base_ability=s.base_ability.at[r,side,z].set(insomnia))
  roots=self.public_roots(t);proof=cb.public_proof(s,t.cursor,roots,yes,self.data);self.assertEqual(proof.known.tolist(),[[True,True]]*8)
  t=t._replace(execution=t.execution._replace(baton_transaction=t.execution.baton_transaction._replace(public=proof)))
  _,p=cb.begin_start(t,s,side,move,move,context,acts=yes,eligible=yes);self.assertEqual(p.valid.tolist(),[True]*8)
  np.testing.assert_array_equal(p.before.status,s.status);np.testing.assert_array_equal(p.before.status_counter,s.status_counter)
  # The same actual healthy/public coherence predicates stay required.
  bad_roots=self.public_roots(t);bad_roots[1].cursor.stream_id=bad_roots[1].cursor.stream_id.at[0].add(1)
  changed=cb.public_proof(s,t.cursor,bad_roots,yes,self.data);bad=t._replace(execution=t.execution._replace(baton_transaction=t.execution.baton_transaction._replace(public=changed)))
  self.assertEqual(cb.begin_start(bad,s,side,move,move,context,acts=yes,eligible=yes)[1].valid.tolist(),[False]+[True]*7)
  for field,value in [('ability',natural),('status',5),('status',1),('status',4),('status_counter',1),('hp',0),('base_ability',self.ability)]:
   bad=s._replace(**{field:getattr(s,field).at[0,side[0],0].set(value)})
   if field=='ability':bad=bad._replace(base_ability=bad.base_ability.at[0,side[0],0].set(value))
   self.assertEqual(cb.begin_start(t,bad,side,move,move,context,acts=yes,eligible=yes)[1].valid.tolist(),[False]+[True]*7,(field,value))
  for caller,acts in((move+1,yes),(move,~yes)):
   self.assertEqual(cb.begin_start(t,s,side,move,caller,context,acts=acts,eligible=yes)[1].valid.tolist(),[False]*8)
  # Literal original before-copy boundary: no private-request/FULL completion
  # is invented here. The coherent cached snapshot is explicitly fixture data.
  s=s._replace(baton_pass=s.baton_pass.at[r,side].set(True));tr=t.execution.baton_transaction;h=tr.history._replace(side=side,private=cb.snapshot(s),boosts=s.boosts)
  t=t._replace(execution=t.execution._replace(baton_transaction=tr._replace(history=h)))
  self.assertEqual(cb.before_copy(t,s,side,z+3,context,eligible=yes)[1].valid.tolist(),[True]*8)
  for field,value in [('status',3),('status_counter',1),('hp',99)]:
   bad=s._replace(**{field:getattr(s,field).at[0,side[0],0].set(value)})
   self.assertEqual(cb.before_copy(t,bad,side,z+3,context,eligible=yes)[1].valid.tolist(),[False]+[True]*7,(field,value))
  # Coherent Natural Cure replacement still refuses, including at before-copy.
  bad=s._replace(ability=s.ability.at[0,side[0],0].set(natural),base_ability=s.base_ability.at[0,side[0],0].set(natural));tr=t.execution.baton_transaction
  changed=t._replace(execution=t.execution._replace(baton_transaction=tr._replace(history=tr.history._replace(private=cb.snapshot(bad)))))
  self.assertEqual(cb.before_copy(changed,bad,side,z+3,context,eligible=yes)[1].valid.tolist(),[False]+[True]*7)
  # Incoming major status and incoming dormant ability keep their old closure.
  for field,value in [('status',2),('ability',insomnia)]:
   bad=s._replace(**{field:getattr(s,field).at[0,side[0],3].set(value)})
   if field=='ability':bad=bad._replace(base_ability=bad.base_ability.at[0,side[0],3].set(value))
   tr=t.execution.baton_transaction;changed=t._replace(execution=t.execution._replace(baton_transaction=tr._replace(history=tr.history._replace(private=cb.snapshot(bad)))))
   self.assertEqual(cb.before_copy(changed,bad,side,z+3,context,eligible=yes)[1].valid.tolist(),[False]+[True]*7)
  # Departure preserves BRN/PSN and their zero clock, rather than transferring
  # them or changing the original TOX-only clock reset prediction.
  expected=cb.departed_private(cb.snapshot(s),side,z)
  np.testing.assert_array_equal(expected.status,s.status);np.testing.assert_array_equal(expected.status_counter,s.status_counter)

 def test_exact_header_namespace_risk_join_legacy_missing_and_diagnostic_fact(self):
  t,s,side,z,yes=self.transaction();r=j.arange(8);insomnia=self.metadata['vocabulary']['abilities'].index('insomnia');natural=self.metadata['vocabulary']['abilities'].index('naturalcure');cap=self.data.outgoing_insomnia_status
  self.assertEqual(cap.dtype,j.int32);self.assertEqual(cap.shape,(3+len(self.data.quiet_abilities),));self.assertEqual(cap[:3].tolist(),[2,3,len(self.data.quiet_abilities)])
  self.assertEqual(cap[3:].tolist(),[int(i==insomnia)for i in range(len(self.data.quiet_abilities))]);self.assertEqual(int(cap[3+natural]),0)
  self.assertEqual(cb._bound(t,self.data).tolist(),[True]*8);risk.check_compiled(self.runtime['kernel_risk_compiled']);self.assertTrue(all(isinstance(x,jax.Array)for x in t.execution.risks.tables))
  np.testing.assert_array_equal(cb._data(t).outgoing_insomnia_status,cap)
  for i in (0,1,2,3+insomnia,len(cap)-1):
   tables=t.execution.risks.tables._replace(baton_outgoing_insomnia_status=cap.at[i].add(1));bad=t._replace(execution=t.execution._replace(risks=t.execution.risks._replace(tables=tables)))
   self.assertEqual(cb._bound(bad,self.data).tolist(),[False]*8)
  for bad in(cap[None,:],cap.astype(j.bool_),cap[:-1]):
   with self.assertRaises(ValueError):cb.shape(self.data._replace(outgoing_insomnia_status=bad))
  s=s._replace(status=s.status.at[r,side,z].set(2),ability=s.ability.at[r,side,z].set(insomnia),base_ability=s.base_ability.at[r,side,z].set(insomnia))
  self.assertEqual(cb._quiet(s,side,z,self.data._replace(outgoing_insomnia_status=None),outgoing_dormant=True).tolist(),[False]*8)
  for i in(0,1,2):self.assertEqual(cb._insomnia_status(s.ability[r,side,z],s.status[r,side,z],z,self.data._replace(outgoing_insomnia_status=cap.at[i].add(1))).tolist(),[False]*8)
  tables=t.execution.risks.tables._replace(baton_outgoing_insomnia_status=j.zeros(0,j.int32));legacy=t._replace(execution=t.execution._replace(risks=t.execution.risks._replace(tables=tables)));self.assertIsNone(cb._data(legacy).outgoing_insomnia_status)
  for name in('engine_binding','source_binding','move_binding','producer_binding'):
   bad=t._replace(**{name:j.broadcast_to(getattr(t,name),(8,8))})
   with self.assertRaises(ValueError):cb._bound(bad,self.data)
  capture=diag.empty(8);t=t._replace(execution=t.execution._replace(diagnostics=capture));move=z+self.move
  t,p=cb.begin_start(t,s,side,move,move,self.runtime['direct_hit_context'],acts=yes,eligible=yes)
  records=diag.refused_rows(t.execution.diagnostics,j.ones(8,bool))['rows'];self.assertEqual(len(records),8)
  for row in records:
   actual=row['baton_start']['records'][0];self.assertEqual(actual['quiet_status_lookup'],1);self.assertIs(actual['guard_values']['quiet_status'],True);self.assertIs(actual['guard_values']['quiet_ability'],True)
  self.assertIsNone(diag.baton_start_guard_values({'status':2,'counter':0,'quiet_status_lookup':None})['quiet_status'])

 def test_original_source_callbacks_generator_namespace_and_unchanged_production(self):
  script=r"""
const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3'),H=x=>require('crypto').createHash('sha256').update(''+x).digest('hex'),rows=[];
const abilities=['_none',...[...new Set(d.abilities.all().filter(x=>x.exists&&x.gen<=3).map(x=>x.id).filter(Boolean))].sort()];
for(const [id,hook]of[['insomnia','onUpdate'],['naturalcure','onSwitchOut']])for(const status of['brn','psn','slp']){
 const words=[],p={status,hp:71,maxhp:100,pp:[4,5,6,7],boosts:{atk:1},clearStatus(){this.status=''},cureStatus(){this.status=''},toString(){return'p1a: Actor'}};
 d.abilities.get(id)[hook].call({add(...a){words.push('|'+a.map(String).join('|'))}},p);
 rows.push({id,hook,status,after:p.status,hp:p.hp,maxhp:p.maxhp,pp:p.pp,boosts:p.boosts,words,sha:H(d.abilities.get(id)[hook])});
}
console.log(JSON.stringify({abilities,rows}));
"""
  value=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True));self.assertEqual(len(value['abilities']),81);self.assertEqual(value['abilities'][26],'insomnia');self.assertEqual(value['abilities'][38],'naturalcure')
  for row in value['rows']:
   self.assertEqual((row['hp'],row['maxhp'],row['pp'],row['boosts']),(71,100,[4,5,6,7],{'atk':1}))
   self.assertEqual(row['sha'],cb.INSOMNIA_UPDATE_SHA256 if row['id']=='insomnia'else cb.NATURAL_CURE_SWITCH_OUT_SHA256)
   if row['id']=='insomnia'and row['status']in('brn','psn'):self.assertEqual(row['words'],[]);self.assertEqual(row['after'],row['status'])
   else:self.assertEqual(row['after'],'');self.assertEqual(len(row['words']),1)
  with patch.object(cb,'INSOMNIA_UPDATE_SHA256','0'*64):
   with self.assertRaisesRegex(ValueError,'dormant outgoing'):cb.compile_data(self.producer,self.metadata,self.rules)
  for name in('transition.py','entry.py','baton.py','state.py','mechanics.py','engine_retained_encoder_driver.py','engine_full_v4_event_device.py','engine_lifecycle_event_device.py'):
   self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+name]),name)
  before=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_baton_pass_callbacks.py'],text=True));after=ast.parse(Path(cb.__file__).read_text());changed={'compile_data','shape','_bound','_data','_ability_quiet','_quiet','_start_diagnostics'}
  functions=lambda tree:{n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name not in changed|{'_insomnia_status'}}
  self.assertEqual(functions(before),functions(after))
  self.assertEqual([n.name for n in ast.walk(after)if isinstance(n,ast.FunctionDef)and n.name in('begin_start','before_copy')],['begin_start','before_copy'])

if __name__=='__main__':unittest.main()
