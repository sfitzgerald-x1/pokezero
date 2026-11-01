"""Only literal maturation/row-commit controls; no project runtime invocation."""
import ast,unittest
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_baton_pass_callbacks as cb
from gpu_engine import engine_retained_encoder_driver as driver,state,private_requests

class Core(NamedTuple):
 baton_history:object
 requests:object
class BatonInactiveHistoryTests(unittest.TestCase):
 def history(self):
  s=state.empty_state(2);z=j.zeros(2,j.int32)
  cursor=SimpleNamespace(stream_id=z+991000,turn=z,generation=j.zeros((2,2),j.int32))
  return cb.empty_history(s,cursor,j.broadcast_to(j.arange(6),(2,2,6)))._replace(side=z,frame=z,public_slot=z,ready=j.ones(2,bool))
 def test_actual_driver_maturation_assignment_preserves_entire_inactive_history(self):
  h=self.history();changed=jax.tree.map(lambda v:~v if v.dtype==j.bool_ else v+1,h)
  core=Core(h,None);frame=SimpleNamespace(callbacks=None);observers=()
  tree=ast.parse(Path('gpu_engine/engine_retained_encoder_driver.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_consume_accounted')
  call=lambda n:isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='baton_callbacks'and n.func.attr=='mature'
  assignment=next(n for n in ast.walk(fn)if isinstance(n,ast.Assign)and any(call(c)for c in ast.walk(n)))
  advance=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='advance')
  consume=next(n for n in ast.walk(advance)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='_consume')
  self.assertEqual(ast.unparse(consume.args[3]),'executed')
  active=j.array([False,True]);spy=SimpleNamespace(mature=lambda *args:changed)
  env=dict(core=core,frame=frame,observers=observers,active=active,baton_callbacks=spy,_rows=driver._rows)
  exec(compile(ast.Module(body=[assignment],type_ignores=[]),'actual-driver-maturity-commit','exec'),env)
  got=env['core'].baton_history
  for before,after,want in zip(jax.tree.leaves(h),jax.tree.leaves(got),jax.tree.leaves(changed),strict=True):
   np.testing.assert_array_equal(after[0],before[0])
   np.testing.assert_array_equal(after[1],want[1])
 def test_empty_start_record_actually_clears_active_ready(self):
  h=self.history();rec=cb.empty_record(h);tr=cb.Transaction(h,None,None,None,None,rec)
  frame=SimpleNamespace(callbacks=SimpleNamespace(execution=SimpleNamespace(baton_transaction=tr)))
  z=j.zeros(2,j.int32);ledger=SimpleNamespace(site=j.full((2,1),39,j.int32),stream_id=h.stream_id,refusal=z,stage=j.full((2,1),3,j.int32),serial=j.zeros((2,1),j.int32))
  pub=SimpleNamespace(cursor=SimpleNamespace(stream_id=h.stream_id),context=SimpleNamespace(tendency=SimpleNamespace(baton=j.ones((2,2),bool))),mons=SimpleNamespace(boosts=j.zeros((2,2,7),j.int32)))
  def request(kind):return SimpleNamespace(actions=SimpleNamespace(admitted=j.ones(2,bool),facts=SimpleNamespace(request=SimpleNamespace(kind=j.full(2,int(kind),j.int32)))))
  q=(request(private_requests.RequestKind.FORCE_SWITCH),request(private_requests.RequestKind.WAIT))
  with patch.object(cb,'owned',return_value=j.zeros(2,bool)):
   self.assertEqual(cb.mature(h,frame,(ledger,ledger),(pub,pub),q).ready.tolist(),[False,False])

if __name__=='__main__':unittest.main()
