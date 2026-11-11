"""Original Liquid Ooze cancellation is unsupported input, not a physics fix."""
import hashlib,json,subprocess,unittest
from unittest import mock
import jax.numpy as j
import numpy as np
from _showdown_root import showdown_root
import test_gpu_engine_leech_seed_owning as fixture
from gpu_engine import engine_leech_seed_callbacks as cb,engine_execution_completeness as ex,sustain

class LeechLiquidOozeRefusalTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  original=fixture.fixture.NoDamageOwningTests.setUpClass
  def original_ooze_namespace():
   original();names=fixture.fixture.NoDamageOwningTests.metadata['vocabulary']['abilities']
   if 'liquidooze'not in names:names.append('liquidooze')
  with mock.patch.object(fixture.fixture.NoDamageOwningTests,'setUpClass',side_effect=original_ooze_namespace):fixture.LeechSeedOwningTests.setUpClass()
  cls.f=fixture.LeechSeedOwningTests;cls.case=cls.f();cls.case.setUp()
 def test_original_callback_and_unhandled_victim_for_half_and_full_hp_recipient(self):
  script="""const d=require(process.argv[1]).Dex.mod('gen3'),f=d.abilities.get('liquidooze').onSourceTryHeal,calls=[];const result=f.call({debug(){},activeMove:null,damage(...x){calls.push(x);}},25,{hp:201,maxhp:201},{id:'victim'},{id:'leechseed'});console.log(JSON.stringify({sha:require('crypto').createHash('sha256').update(f.toString()).digest('hex'),result,calls}));"""
  witness=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root()/'dist/sim/dex.js')],text=True));self.assertEqual(witness,dict(sha=cb.LIQUID_OOZE_SHA,result=0,calls=[[25,None,None,None,True]]))
  s,seeded,source,target,eligible,t,c,_,_=self.case.component();rows=j.arange(8);sp=seeded.active[rows,source];tp=seeded.active[rows,target];ooze=self.f.metadata['vocabulary']['abilities'].index('liquidooze');self.assertTrue(bool(self.f.rules.device.ability_supported[ooze]));self.assertEqual(int(self.f.helper.device.leech_seed.liquid_ooze_id),ooze)
  blocked=seeded._replace(ability=seeded.ability.at[rows,target,tp].set(ooze));after,tape,drain,heal=self.case.residual(blocked,source,target,t)
  self.assertFalse(bool(cb.captured_matches(tape,drain).any()));self.assertFalse(bool(cb.captured_matches(tape,heal).any()));self.assertTrue(bool(((tape.refusal&256)!=0).all()))
  ledger=tape.execution.ledger;self.assertEqual(ledger.count.tolist(),[3]*8)
  self.assertEqual(ledger.site[:,:3].tolist(),[[int(ex.Site.LEECH_SEED),int(ex.Site.LEECH_DRAIN),int(ex.Site.LEECH_HEAL)]]*8);self.assertEqual(ledger.stage[:,:3].tolist(),[[2,1,1]]*8);self.assertTrue(bool(((ledger.refusal&ex.UNKNOWN_SITE)!=0).all()));self.assertEqual(tape.count.tolist(),[2]*8)
  # The unchanged, known-incomplete numerical helper still heals; nothing from
  # that incompatible result is published or receives a stage2 waiver.
  original=sustain.leech(blocked.hp[rows,target,tp],blocked.max_hp[rows,target,tp],blocked.volatile[rows,target,tp],blocked.hp[rows,source,sp],blocked.max_hp[rows,source,sp],eligible=j.ones(8,bool))
  np.testing.assert_array_equal(after.hp[rows,target,tp],original.hp);np.testing.assert_array_equal(after.hp[rows,source,sp],original.recipient_hp)
  self.assertEqual(after.hp[rows,source,sp][1::2].tolist(),[201]*4)
 def test_recipient_ooze_does_not_borrow_victim_hook_and_ability_drift_refuses(self):
  s,seeded,source,target,eligible,t,c,_,_=self.case.component();rows=j.arange(8);sp=seeded.active[rows,source];tp=seeded.active[rows,target];ooze=self.f.metadata['vocabulary']['abilities'].index('liquidooze')
  recipient=seeded._replace(ability=seeded.ability.at[rows,source,sp].set(ooze));after,tape,drain,heal=self.case.residual(recipient,source,target,t);self.assertEqual(cb.captured_matches(tape,heal).tolist(),[True]*8)
  tape,p=cb.begin_drain(t,seeded,target,self.f.runtime['direct_hit_context'],eligible=j.ones(8,bool));result=sustain.leech(seeded.hp[rows,target,tp],seeded.max_hp[rows,target,tp],seeded.volatile[rows,target,tp],seeded.hp[rows,source,sp],seeded.max_hp[rows,source,sp],eligible=j.ones(8,bool));mutated=seeded._replace(hp=seeded.hp.at[rows,target,tp].set(result.hp),ability=seeded.ability.at[rows,target,tp].set(ooze));tape,cap=cb.finish_drain(tape,p,mutated,result)
  self.assertFalse(bool(cb.captured_matches(tape,cap).any()));self.assertEqual(tape.execution.ledger.stage[:,1].tolist(),[1]*8)
if __name__=='__main__':unittest.main()
