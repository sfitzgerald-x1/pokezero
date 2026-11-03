import unittest
from dataclasses import replace
from types import SimpleNamespace
from pokezero import rollout_truth as t
from gpu_engine import device_rollout_truth as producer
from gpu_engine.device_rollout_truth_score import backup_from_grid

SEL=dict(stream=7,frame=4,owner=1)

def full(value):return [dict(replicate=i,status='COMPLETE',outcome=value,requests=3,reason=None)for i in range(64)]

class MetricsTests(unittest.TestCase):
 def test_real_serialized_vocabulary_has_computed_not_stored_digest(self):
  from dataclasses import asdict
  from pokezero.v5_contract import LabelVocabulary
  vocabulary=LabelVocabulary('a'*64,('tackle',),(('ditto',1),))
  metadata=asdict(vocabulary)
  self.assertNotIn('sha256',metadata)
  self.assertEqual(producer.label_vocabulary_sha256(metadata),vocabulary.sha256)
 def test_recipe_is_fixed_by_branch_and_replicate_not_head_or_result(self):
  r=t.Recipe(2103300000)
  a=r.keys(SEL,1,2,'a'*64)
  self.assertEqual(a,t.Recipe(2103300000).keys(dict(SEL),1,2,'a'*64));self.assertEqual(len(a),64)
  self.assertEqual(len(set(a)),64);self.assertNotEqual(a,r.keys(SEL,2,2,'a'*64))
  with self.assertRaises(ValueError):t.Recipe(1,rollouts=63)
 def test_no_survivor_renormalization_and_nonterminal_outcome_refuses(self):
  r=full(1);r[0]=dict(replicate=0,status='CAPPED',outcome=None,requests=9,reason='cap')
  s=t.summarize_rollouts(r);self.assertIsNone(s['value']);self.assertEqual(s['upper'],1);self.assertEqual(s['lower'],62/64)
  r[0]['outcome']=1
  with self.assertRaises(ValueError):t.summarize_rollouts(r)
  with self.assertRaises(ValueError):t.summarize_rollouts(r[:-1])
 def test_brier_uses_terminal_distribution_not_squared_mean(self):
  c=t.calibration([0],[[1,-1]*32]);self.assertEqual(c['brier'],.25);self.assertEqual(c['ece'],0)
  self.assertAlmostEqual(t.spearman([1,2,2,4],[1,3,3,4]),1)
  self.assertIsNone(t.spearman([1,1],[2,2]))
 def test_actual_missing_root_and_child_stay_in_denominator(self):
  roots=[dict(selector=SEL,prediction=1.,outcome=1,backup=1.),dict(selector=dict(stream=8,frame=4,owner=0),prediction=None,outcome=None)]
  kids=[dict(selector=SEL,action=0,prediction=1.,rollouts=full(1)),dict(selector=SEL,action=1,prediction=-1.,rollouts=full(-1))]
  m=t.metrics(roots,kids)
  self.assertEqual(m['root_unknown'],1);self.assertEqual(m['top_two']['unknown_groups'],1)
  self.assertEqual(m['top_two']['resolution'],1);self.assertEqual(m['decision'],'UNKNOWN_INCOMPLETE_DENOMINATOR')
  kids[0]['rollouts']=producer.missing(t.Recipe(3),'missing')
  m=t.metrics(roots,kids);self.assertEqual(m['child_unknown'],1);self.assertEqual(m['child_known'],1)
 def test_top_two_truth_rank_gap_and_backup_must_improve(self):
  roots=[dict(selector=SEL,prediction=0.,outcome=1,backup=1.)]
  kids=[dict(selector=SEL,action=0,prediction=1.,rollouts=full(1)),dict(selector=SEL,action=1,prediction=-1.,rollouts=full(-1))]
  m=t.metrics(roots,kids);self.assertEqual(m['decision'],'PASS');self.assertTrue(m['bars']['backup_lower'])
  self.assertEqual(m['top_two']['gap_units'],'expected_outcome[-1,1]');self.assertEqual(m['top_two']['primary_gap'],.05)
  self.assertTrue(m['top_two_probability_gap_05_sensitivity']['not_a_bar'])
  kids[0]['prediction'],kids[1]['prediction']=-1.,1.
  self.assertEqual(t.metrics(roots,kids)['decision'],'FAIL')
 def test_game_clustered_split_never_splits_frames_or_children(self):
  selectors=[dict(stream=i,frame=f,owner=i%2)for i in range(200)for f in (0,4,12,24)]
  p=t.partition(selectors,seed=500)
  self.assertEqual(len(p),200);self.assertEqual(set(p.values()),{'train','validation','test'})
  self.assertEqual(p,t.partition(list(reversed(selectors)),seed=500))
 def test_all_replies_backup_no_refusal_reweight_and_terminal_value(self):
  rows=[dict(action=a,reply=r,weight=.25 if r==0 else.75,status='OBSERVED',terminal_value=None)for a in (0,1)for r in (0,1)]
  g=dict(status='OBSERVED',rows=rows,own_actions=[0,1],replies=[0,1],reply_weights=[.25,.75])
  self.assertAlmostEqual(backup_from_grid(g,[1.,-1.,0.,1.]),.75)
  self.assertIsNone(backup_from_grid(g,[1.,None,0.,1.]))
  rows[0].update(status='COMPLETE',terminal_value=-1)
  self.assertAlmostEqual(backup_from_grid(g,[1.,-1.,0.,1.]),.75)
  rows[2]['weight']=.5
  with self.assertRaises(ValueError):backup_from_grid(g,[1.,-1.,0.,1.])
 def test_root_acceptance_requires_both_digests_and_literal_public(self):
  class Bank:
   _host=staticmethod(lambda x:x)
   _observation=staticmethod(lambda x:x)
  s={'hp':[2]};p={'revealed':[3]};v={'numeric':[.1]}
  world=t.digest(s);root=t.digest(dict(selector=SEL,world=world,public=p))
  position=dict(world_id=world,root_id=root,observation=v)
  self.assertTrue(producer.accept_root(s,p,v,SEL,position,Bank,t))
  for k in ('world_id','root_id','observation'):
   bad=dict(position);bad[k]='foreign'
   self.assertFalse(producer.accept_root(s,p,v,SEL,bad,Bank,t))
 def test_disabled_collection_does_not_load_inputs_or_original_modules(self):
  with self.assertRaises(ValueError):producer.collect(None,None,None)


class InertContinuationTests(unittest.TestCase):
 def test_actual_batch_controller_both_owners_cap_refusal_and_terminal(self):
  # Only array/control operations; every model/chance/transition callback here
  # is inert. No real engine step or actual forward is called.
  import jax
  import jax.numpy as j
  import numpy as np
  from typing import NamedTuple
  class S(NamedTuple):done:object;winner:object;keys:object;turns:object
  class Step(NamedTuple):state:object;invalid_actions:object
  class Out(NamedTuple):step:object;trackers:object
  source=S(j.asarray([False]),j.asarray([0]),j.zeros((1,2),j.uint32),j.asarray([0]))
  actor=object.__new__(producer.RawExecutor);seen=[]
  actor.policy_keys=lambda p,n:p[:,None,:].repeat(2,axis=1)
  actor.chance=lambda s:j.zeros((len(s.done),),j.int32)
  def query(s,p,k):
   seen.append((np.asarray(s.keys).copy(),np.asarray(s.done).copy()))
   valid=np.ones(len(s.done),bool);valid[0]=False
   return None,None,j.zeros((len(s.done),2),j.int32),j.asarray(valid)
  actor.query=query
  def advance(s,p,a,c):
   ticks=s.turns+1
   # First child completes with seat0 winner; second child reaches cap.
   complete=j.arange(len(s.done))<64
   return Out(Step(s._replace(done=s.done|complete,winner=j.zeros_like(s.winner),turns=ticks),j.zeros(len(s.done),bool)),p)
  actor.advance=advance
  kids=[dict(state=source,public=j.zeros((1,1)),selector=dict(SEL,owner=owner),action=0,reply=1,chance_id='a'*64)for owner in (0,1)]
  rows=actor.continuation_batch(kids,t.Recipe(3,max_requests=2))
  self.assertEqual(len(rows),2);self.assertEqual(len(seen[0][0]),2048)
  self.assertTrue(seen[0][1][128:].all())
  self.assertEqual(rows[0][0]['status'],'REFUSED');self.assertEqual(rows[0][1]['outcome'],1)
  self.assertTrue(all(r['status']=='CAPPED'and r['outcome']is None for r in rows[1]))
  self.assertTrue(np.array_equal(np.asarray(source.keys),np.zeros((1,2),np.uint32)))
  self.assertTrue(np.asarray(seen[1][1])[:64].all())
  seen.clear()
  recipe=t.Recipe(3,max_requests=2)
  mixed=actor.continuation_batch([None,kids[1],None],recipe)
  self.assertIsNone(mixed[0]);self.assertIsNone(mixed[2]);self.assertEqual(len(mixed[1]),64)
  self.assertTrue(seen[0][1][:64].all());self.assertFalse(seen[0][1][64:128].any());self.assertTrue(seen[0][1][128:].all())
  declared=recipe.keys(kids[1]['selector'],0,1,'a'*64)
  expected=np.asarray(jax.vmap(jax.random.PRNGKey)(j.asarray([x[0]for x in declared],j.uint32)))
  self.assertTrue(np.array_equal(seen[0][0][64:128],expected))

class ManifestTests(unittest.TestCase):
 def test_actual_counts_are_reported_even_when_human_brief_differs(self):
  import tempfile,json,hashlib
  from pathlib import Path
  from types import SimpleNamespace
  from unittest.mock import Mock
  declaration={'path':'/immutable/declaration','sha256':'b'*64}
  config=dict(declaration=declaration,positions={},siblings={})
  positions=[dict(status='OBSERVED'if i<1000 else'REFUSED',terminal=dict(status='COMPLETE'))for i in range(1024)]
  groups=[dict(branches=[{}]if i<900 else[])for i in range(1024)]
  header=dict(device_bank=declaration)
  decl=dict(source_commit='a'*40,source_sha256='c'*64,pool_sha256='d'*64,actors=[])
  fake=SimpleNamespace(POSITIONS='p',SIBLINGS='s',declaration=lambda ref:(decl,None,range(256),SimpleNamespace(sha256='e'*64)),
   _records=lambda ref,kind:(header,positions if kind=='p'else groups),resolve_device_bank=lambda *a:None)
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'index.json';p.write_text(json.dumps(dict(declaration=declaration,namespace=dict(source_commit='a'*40),cost_roots=[])))
   config['root_index']=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
   manifest=producer.bank_manifest(config,fake)[-1]
  self.assertEqual(manifest['declared_slots'],1024);self.assertEqual(manifest['declared_children'],900)
  self.assertEqual(manifest['brief_comparison']['actual_minus_reported_children'],900-8378)
 def test_all_five_test_folds_are_disjoint_from_fit_and_validation(self):
  selectors=[dict(stream=i,frame=f,owner=i%2)for i in range(256)for f in (0,4,12,24)]
  test_counts={i:0 for i in range(256)}
  for k in range(5):
   p=t.partition(selectors,seed=42,holdout_fold=k)
   for i,r in p.items():test_counts[i]+=int(r=='test')
  self.assertEqual(set(test_counts.values()),{1})

class ReconstructionControllerTests(unittest.TestCase):
 def test_actual_prefix_controller_stops_each_row_at_last_needed_selector(self):
  import jax
  import jax.numpy as j
  import numpy as np
  from typing import NamedTuple
  from unittest.mock import patch
  from gpu_engine import transition,v5_model,engine_request_source
  class S(NamedTuple):done:object;counter:object
  class View(NamedTuple):numeric:object
  class Init(NamedTuple):state:object;trackers:object
  class Step(NamedTuple):state:object;invalid_actions:object
  class Out(NamedTuple):step:object;trackers:object
  counter=[];published=[]
  def query(s,p,streams,frames,params,seed):
   view=View(s.counter[:,None])
   return dict(input_valid=j.ones(16,bool),requested=j.ones((16,2),bool),observations=(view,view),actions=j.zeros((16,2),j.int32),policy_logits=j.zeros((16,2,9)),policy_finite=j.ones((16,2),bool),physical=j.zeros((16,2),j.int32),chance=j.zeros(16,j.int32))
  def advance(s,p,a,c):
   active=~np.asarray(s.done);counter.append(active.copy())
   return Out(Step(s._replace(counter=s.counter+(~s.done).astype(j.int32)),j.zeros(16,bool)),p)
  fakebank=SimpleNamespace(make_executor=lambda *a:SimpleNamespace(query=query,advance=advance),
   _one=lambda x,row:jax.tree.map(lambda a:a[row:row+1],x),_one_view=lambda x,row:View(x.numeric[row:row+1]))
  snapshots=SimpleNamespace(SOURCE_FIELDS=('observations','requested','policy_logits','policy_finite','actions'),save=lambda path,context,fields,**kw:(published.append((int(context.streams[0]),int(context.frames[0])))or dict(path=str(path),sha256='a'*64)))
  positions=[]
  for r in range(16):
   for frame in (0,4,12,24):positions.append(dict(selector=dict(stream=r,frame=frame,owner=0),status='OBSERVED'if r or frame==0 else'REFUSED'))
  batch=SimpleNamespace(games=[SimpleNamespace(team_indices=(2*i,2*i+1),stream_id=i)for i in range(16)],constructor_seed=1,sampling_seed=2,gender_seed=3,gender_domain=4,gender_root_ids=range(16))
  decl=dict(actors=[dict(archive=dict(path='unused',sha256='a'*64),checkpoint_sha256='b'*64)],category_vocabulary_sha256='c'*64,label_vocabulary=dict(source_binding_sha256='d'*64,moves=('tackle',),species_variants=(('ditto',1),)))
  with patch.object(v5_model,'load',return_value=(None,None,None)),patch.object(transition,'state_from_pool',return_value=S(j.zeros(16,bool),j.zeros(16,j.int32))),patch.object(transition,'initialize_with_public',side_effect=lambda s,*a,**k:Init(s,j.zeros((16,1)))),patch.object(engine_request_source,'_gender',side_effect=lambda s,*a:s),patch.object(producer,'accept_root',return_value=True):
   entries=producer.reconstruct(SimpleNamespace(engine_tables=None,observation_tables=SimpleNamespace(belief=None)),SimpleNamespace(arrays={}),None,decl,[batch],positions,dict(namespace={}),__import__('pathlib').Path('/unused'),fakebank,SimpleNamespace(FRAMES=(0,4,12,24)),snapshots)
  self.assertEqual(len(counter),24);self.assertTrue(all(not a[0]for a in counter))
  self.assertEqual([x for x in published if x[0]==0],[(0,0)])
  self.assertEqual(sum(e['status']=='SAVED'for e in entries),61)
  self.assertEqual(len(entries),64)

if __name__=='__main__':unittest.main()
