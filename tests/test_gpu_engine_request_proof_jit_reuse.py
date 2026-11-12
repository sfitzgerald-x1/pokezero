"""Source/fake controls only: no project imports, backend, models, or games."""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
import weakref
BASE='0d51ac06e34af37ad868850456c5209b2b045e29'
FEATURE='a6c75ad734a024825afbdd3d9f09b867ae405cd7'
INTEGRATED='b8092a85d9c1f2ed9c0daf4c4bd8e833a94a2380'
SOURCE=Path('gpu_engine/engine_retained_encoder_driver.py')

def tree():return ast.parse(SOURCE.read_text())
def tree_at(ref):return ast.parse(subprocess.check_output(['git','show',ref+':'+str(SOURCE)],text=True))
def function(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)

class RequestProofReuseTests(unittest.TestCase):
 def test_once_prepared_four_owner_functions_fresh_dynamic_inputs_and_refusals(self):
  wrappers=[];calls=[]
  def jit(fn):
   def invoke(*args):calls.append((fn,args));return fn(*args)
   wrappers.append(invoke);return invoke
  ns=dict(jax=SimpleNamespace(jit=jit),_REQUEST_PROOF_ORIGINS=weakref.WeakKeyDictionary());exec(compile(ast.Module([function(tree(),'_prepare_request_proof_pair')],[]),'<actual preparation factory>','exec'),ns)
  static=object()
  def expected(engine,before,public,*,owner,tape,healthy,data):
   self.assertIs(data,static)
   return ('CURRENT'if healthy and public['generation']==tape['generation']else'UNKNOWN',owner,engine,before,public['encounter'])
  types=ns['_prepare_request_proof_pair'](expected,data=static);encore=ns['_prepare_request_proof_pair'](expected,data=static)
  self.assertEqual(len(wrappers),4);self.assertEqual(calls,[])
  for step in range(4):
   engine=object();before=object();tape=dict(generation=step)
   for pair in (types,encore):
    for owner,proof in enumerate(pair):
     public=dict(generation=step,encounter=step+owner);actual=proof(engine,before,public,tape,True)
     self.assertEqual(actual,expected(engine,before,public,owner=owner,tape=tape,healthy=True,data=static));self.assertIs(actual[2],engine);self.assertIs(actual[3],before)
     self.assertEqual(proof(engine,before,public,tape,False)[0],'UNKNOWN')
     self.assertEqual(proof(engine,before,dict(public,generation=step-1),tape,True)[0],'UNKNOWN')
  self.assertEqual(len(wrappers),4)

 def test_exception_does_not_cache_dynamic_result(self):
  calls=[]
  def expected(engine,before,public,*,owner,tape,healthy):
   calls.append((engine,owner))
   if not healthy:raise ValueError('literal current proof refuses')
   return engine,owner
  ns=dict(jax=SimpleNamespace(jit=lambda fn:fn),_REQUEST_PROOF_ORIGINS=weakref.WeakKeyDictionary());exec(compile(ast.Module([function(tree(),'_prepare_request_proof_pair')],[]),'<actual preparation factory>','exec'),ns);pair=ns['_prepare_request_proof_pair'](expected);first=object();second=object()
  with self.assertRaises(ValueError):pair[0](first,None,None,None,False)
  self.assertEqual(pair[0](second,None,None,None,True),(second,0));self.assertEqual(calls,[(first,0),(second,0)])

 def test_original_outer_guards_callbacks_barriers_session_commit_unchanged(self):
  # The immutable feature-era proof stays separate from the integrated source.
  live=tree();current=tree_at(FEATURE);old=tree_at(BASE);integrated=tree_at(INTEGRATED)
  for name in ('_checked','_checked_witness','_session','_timed','_consume','_consume_accounted','advance','prepare'):
   self.assertEqual(dump(function(live,name)),dump(function(integrated,name)),name)
  for name in ('_checked','_checked_witness','_session','_timed','_consume','_consume_accounted'):
   self.assertEqual(dump(function(current,name)),dump(function(old,name)),name)
  advance=deepcopy(function(current,'advance'));baseline=function(old,'advance');original_requests=next(n for n in ast.walk(baseline)if isinstance(n,ast.FunctionDef)and n.name=='requests_phase')
  class RestoreRequests(ast.NodeTransformer):
   def visit_FunctionDef(self,node):
    if node.name=='requests_phase':return deepcopy(original_requests)
    return self.generic_visit(node)
  self.assertEqual(dump(RestoreRequests().visit(advance)),dump(baseline))
  preparation=deepcopy(function(current,'prepare'));baseline=function(old,'prepare')
  def is_new_statement(n):
   if isinstance(n,ast.Assign):return any(isinstance(t,ast.Name)and t.id in ('request_type_proofs','request_encore_proofs')for t in n.targets)
   return isinstance(n,ast.If)and any(isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='_prepare_request_proof_pair'for x in ast.walk(n))
  preparation.body=[n for n in preparation.body if not is_new_statement(n)]
  for n in ast.walk(preparation):
   if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='dict':n.keywords=[k for k in n.keywords if k.arg not in ('request_type_proofs','request_encore_proofs')]
  self.assertEqual(dump(preparation),dump(baseline))

 def test_exact_owner_call_inputs_and_original_proof_bodies(self):
  t=tree();advance=function(t,'advance');requests=next(n for n in ast.walk(advance)if isinstance(n,ast.FunctionDef)and n.name=='requests_phase')
  calls=[n for n in ast.walk(requests)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='proof']
  self.assertEqual(len(calls),2)
  for call in calls:self.assertEqual([ast.unparse(x)for x in call.args],['frame.state','core.engine','observer.retained','frame.callbacks','observer.retained_healthy'])
  preparation=function(t,'prepare');pairs=[n for n in ast.walk(preparation)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='_prepare_request_proof_pair']
  self.assertEqual([ast.unparse(x.args[0])for x in pairs],['color.expected_owned_types','encore_callbacks.expected_request'])
  for name in ('engine_color_change_callbacks.py','engine_encore_callbacks.py','transition.py','v4_encoder.py','engine_kernel_frame_risk.py'):
   # Preserve the feature-era unchanged-body proof and current drift guard.
   self.assertEqual(subprocess.check_output(['git','show',FEATURE+':gpu_engine/'+name]),subprocess.check_output(['git','show',BASE+':gpu_engine/'+name]))
   # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components; the color_change, encore and kernel_frame_risk sources:
   # byte-exact against reviewed SOURCE_PINS in transition_observer_wiring's source-binding subtests.
   if name=='v4_encoder.py':self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show',INTEGRATED+':gpu_engine/'+name]))

 def test_actual_current_call_disabled_predicates_match_original(self):
  old=ast.parse(subprocess.check_output(['git','show',BASE+':'+str(SOURCE)],text=True));new=tree()
  def fragment(t):
   phase=next(n for n in ast.walk(function(t,'advance'))if isinstance(n,ast.FunctionDef)and n.name=='requests_phase');inside=False;body=[]
   for n in phase.body:
    target=n.targets[0].id if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)else None
    if target=='current_types':inside=True
    if target=='requests':break
    if inside:body.append(deepcopy(n))
   return ast.Module(body,[])
  context=SimpleNamespace(data=SimpleNamespace(color_change=object(),encore=object()))
  for mode,current in [('diagnostic',context),('off',context),('kernel-event',None),('kernel-event',SimpleNamespace(data=SimpleNamespace(color_change=None,encore=None)))]:
   def never(*args):raise AssertionError('disabled capability called retained proof')
   r=dict(direct_hit_mode=mode,direct_hit_context=current,request_type_proofs=(never,never),request_encore_proofs=(never,never))
   for t in (old,new):
    ns=dict(r=r);exec(compile(fragment(t),'<actual disabled request proof phase>','exec'),ns);self.assertEqual((ns['current_types'],ns['current_encore']),((None,None),(None,None)))
  def predicates(t):
   phase=next(n for n in ast.walk(function(t,'advance'))if isinstance(n,ast.FunctionDef)and n.name=='requests_phase')
   return [dump(n.test)for n in phase.body if isinstance(n,ast.If)and 'direct_context' in ast.unparse(n.test)]
  self.assertEqual(predicates(old),predicates(new))

 def test_enabled_original_context_function_and_static_identities_mandatory(self):
  ns=dict(jax=SimpleNamespace(jit=lambda fn:fn),_REQUEST_PROOF_ORIGINS=weakref.WeakKeyDictionary())
  nodes=[function(tree(),n)for n in ('_prepare_request_proof_pair','_checked_request_proof_pair')];exec(compile(ast.Module(nodes,[]),'<actual proof origin validators>','exec'),ns)
  context=object();data=object();categories=object();tables=object()
  def expected(*args,**kwargs):return kwargs['owner']
  pair=ns['_prepare_request_proof_pair'](expected,_context=context,data=data,categories=categories,type_tables=tables);check=ns['_checked_request_proof_pair']
  self.assertIs(check(pair,expected,context,data=data,categories=categories,type_tables=tables),pair)
  for changed in ('context','data','categories','type_tables','function'):
   values=dict(context=context,data=data,categories=categories,type_tables=tables,function=expected);values[changed]=object()
   with self.subTest(changed=changed),self.assertRaises(ValueError):check(pair,values['function'],values['context'],data=values['data'],categories=values['categories'],type_tables=values['type_tables'])
  with self.assertRaises(ValueError):check(tuple(reversed(pair)),expected,context,data=data,categories=categories,type_tables=tables)
  with self.assertRaises(ValueError):check((lambda *a:None,lambda *a:None),expected,context,data=data,categories=categories,type_tables=tables)
  self.assertEqual([fn(None,None,None,None,True)for fn in pair],[0,1])

if __name__=='__main__':unittest.main()
