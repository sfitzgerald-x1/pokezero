"""Root independent same-call receipt mutants; no engine/game/model/FULL run."""
import unittest
import jax.numpy as j
import test_gpu_engine_heal_bell_owning as authored
from gpu_engine import engine_heal_bell_callbacks as cb,engine_public_events as public,effects

class RootChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):authored.HealBellTests.setUpClass()
 def fixture(self):
  t=authored.HealBellTests();s,tape,_,actor=t.component(consume_initial=False);b=8;rows=j.arange(b);yes=j.ones(b,bool);move=j.full(b,int(t.helper.device.heal_bell.move),j.int32)
  tape=public.move(tape,actor,s.active[rows,actor],s.active[rows,1-actor],move,eligible=yes)
  tape,p=cb.begin(tape,s,actor,move,t.runtime['direct_hit_context'],t.table,eligible=yes,primary=yes,secondary=yes)
  after,mask=effects.cure_team(s,actor,t.table.effects,eligible=yes,_return_write_mask=True)
  done,c=cb.finish(tape,p,after,mask)
  self.assertTrue(bool(cb.captured_matches(done,c).all()))
  return done,c
 def test_cached_words_cannot_claim_uncured_dead_member_instead_of_actual_cure(self):
  done,c=self.fixture();r=0;slot=int(c.pending.ticket.slot[r]);start=int(done.execution.ledger.start[r,slot]);facts=start+1;count=start+2
  self.assertGreater(int(done.words[r,facts,8]),0);self.assertGreater(int(done.words[r,facts,6]),0);self.assertEqual(int(done.words[r,facts,7]),0)
  # Forge one raw member AND the copied ledger consistently. Same actual
  # Pending.before, after-state and true write-mask still prove a live cure.
  words=done.words.at[r,facts,8].set(0).at[r,facts,7].set(done.words[r,facts,6])
  words=words.at[r,count,7].set(words[r,count,6]).at[r,count,9].set(words[r,count,8])
  origin=start+3;words=words.at[r,origin,7].set(words[r,origin,6])
  ledger=done.execution.ledger._replace(words=done.execution.ledger.words.at[r].set(words[r]))
  forged=done._replace(words=words,execution=done.execution._replace(ledger=ledger))
  self.assertFalse(bool(cb.captured_matches(forged,c)[r]))


class SameCallJoinChecks(RootChecks):
 def test_complete_payload_stamps_and_captured_facts_are_row_local(self):
  import jax
  import numpy as np
  done,c=self.fixture();rows=j.arange(8);r=0;slot=int(c.pending.ticket.slot[r]);start=int(c.pending.start[r])
  matches=jax.jit(lambda tape:cb.captured_matches(tape,c));self.assertTrue(bool(matches(done).all()))
  # Mirror raw mutants into the cached snapshot: this must not turn different
  # header/member facts into the actual captured before/after cure.
  for offset in range(4):
   for column in range(10):
    words=done.words.at[r,start+offset,column].add(1)
    ledger=done.execution.ledger._replace(words=done.execution.ledger.words.at[r].set(words[r]))
    bad=done._replace(words=words,execution=done.execution._replace(ledger=ledger))
    answer=matches(bad);self.assertFalse(bool(answer[r]),(offset,column));self.assertTrue(bool(answer[1:].all()))
  words=done.words
  # Reorder two entire bench member triples consistently in raw/cache. The
  # standalone unique-member grammar is insufficient; actual source order wins.
  first=words[r,start+4:start+7];second=words[r,start+7:start+10]
  words=words.at[r,start+4:start+7].set(second).at[r,start+7:start+10].set(first)
  bad=done._replace(words=words,execution=done.execution._replace(ledger=done.execution.ledger._replace(words=done.execution.ledger.words.at[r].set(words[r]))))
  self.assertFalse(bool(matches(bad)[r]));self.assertTrue(bool(matches(bad)[1:].all()))
  for field in ('turn','phase','generation','start'):
   value=getattr(c.pending,field);value=value.at[r].add(1)
   bad=c._replace(pending=c.pending._replace(**{field:value}));answer=cb.captured_matches(done,bad)
   self.assertFalse(bool(answer[r]),field);self.assertTrue(bool(answer[1:].all()))
  for bad in(c._replace(write_mask=c.write_mask.at[r,1].set(False)),
   c._replace(after=c.after._replace(hp=c.after.hp.at[r,0,0].add(1))),
   c._replace(pending=c.pending._replace(before=c.pending.before._replace(ability=c.pending.before.ability.at[r,0,0].add(1)))),
   c._replace(pending=c.pending._replace(ticket=c.pending.ticket._replace(frame=c.pending.ticket.frame.at[r].add(1))))):
   answer=cb.captured_matches(done,bad);self.assertFalse(bool(answer[r]));self.assertTrue(bool(answer[1:].all()))
  self.assertEqual(done.refusal.tolist(),[public.UNINSTRUMENTED]*8)
