"""Ordered/capped native rejection is the same completion kernel, not forcing."""
import json
from pathlib import Path
import random
import shutil
import subprocess
import unittest
from types import SimpleNamespace

from pokezero.mcts_eval.paper_reference import ReferenceRefusal, SamplingDeadlineExceeded
from pokezero.mcts_eval.paper_reference_sampling import KnownSetTraits, PaperHiddenTeamSampler
from pokezero.mcts_eval.paper_reference_native_membership import draw_native_membership
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
from tests.test_paper_reference_membership_first import receipt
from tests.test_paper_reference_sampler import Generator, set_row
from _showdown_root import requires_showdown, showdown_root
from pokezero.local_showdown import LocalShowdownEnv, LocalShowdownConfig
from pokezero.randbat import load_gen3_randbat_source_cached


class NativeMembershipTests(unittest.TestCase):
    def test_explicit_opt_in_and_defaults(self):
        self.assertEqual(ShowdownWorkerFactory('c','d','e','s').native_membership_batch_size, 0)
        for bad in (True, -1, 17, '8', 8.):
            with self.assertRaises(ReferenceRefusal):
                ShowdownWorkerFactory('c','d','e','s', native_membership_batch_size=bad)
        with self.assertRaises(ReferenceRefusal):
            ShowdownWorkerFactory('c','d','e','s', native_membership_batch_size=8)
        self.assertEqual(ShowdownWorkerFactory('c','d','e','s', history_particles=32,
            membership_first=True, native_membership_batch_size=8).native_membership_batch_size, 8)

    def test_lookahead_preserves_consumed_rng_and_known_set_seeds(self):
        class Fake(Generator):
            def generate_unknown_membership(self, **kw):
                return dict(status='ACCEPTED', consumed=1, completeProposals=1,
                    membershipRejections=0, partySeeds=[kw['seeds'][0]],
                    unknown=[set_row(s) for s in
                        ('Pikachu','Charizard','Blastoise','Venusaur','Gengar')])
        for batch in (1, 8, 16):
            g=Fake(); s=PaperHiddenTeamSampler(g,set_source=SimpleNamespace(species_metadata={}, move_metadata={}))
            rng=random.Random(11); expected=random.Random(11)
            party=expected.getrandbits(32); known_seed=expected.getrandbits(32)
            draw=draw_native_membership(s,(KnownSetTraits('Snorlax'),),rng,
                required=frozenset({'snorlax'}),check=lambda:None,receipt=receipt(),
                batch_size=batch,deadline_at=None)
            self.assertEqual(draw.unknown_party_seeds,(party,))
            self.assertEqual(draw.known[0].seeds,(known_seed,))
            self.assertEqual(rng.getstate(),expected.getstate())

    def test_expired_batch_never_materializes_known_or_accepts_world(self):
        class Fake(Generator):
            def generate_unknown_membership(self, **kw):
                return dict(status='DEADLINE_CANCELLED', consumed=1, completeProposals=1,
                    membershipRejections=0, partySeeds=[], unknown=[])
        g=Fake(); s=PaperHiddenTeamSampler(g,set_source=SimpleNamespace(species_metadata={}, move_metadata={}))
        count=0
        def check():
            nonlocal count
            count+=1
            if count==3: raise SamplingDeadlineExceeded('authoritative clock')
        with self.assertRaises(SamplingDeadlineExceeded):
            draw_native_membership(s,(KnownSetTraits('Snorlax'),),random.Random(11),
                required=frozenset({'snorlax'}),check=check,receipt=receipt(),batch_size=8,deadline_at=None)
        self.assertEqual(g.calls, [])

    def test_pure_native_kernel_counterexamples(self):
        node=shutil.which('node')
        if node is None: self.skipTest('node unavailable')
        module=(Path(__file__).resolve().parents[1]/'scripts/battle_bridge_membership.mjs').as_uri()
        script = """
import assert from 'node:assert/strict';
import {rejectUnknownMembership as run, canonicalPartySpecies as canon} from MODULE;
const rows = names => names.map(species => ({species}));
const args = {seeds:Array.from({length:160},(_,i)=>i), known:['snorlax'],
  required:['murkrow'], maxProposals:8, budgetMs:null};
const rejected=rows(['Pikachu','Charizard','Blastoise','Venusaur','Gengar','Murkrow']);
const accepted=rows(['Murkrow','Charizard','Blastoise','Venusaur','Gengar','Pikachu']);
const out=run(args, seed => seed===4 ? accepted : rejected);
assert.equal(out.consumed,5); assert.equal(out.completeProposals,5);
assert.equal(out.membershipRejections,4); assert.deepEqual(out.partySeeds,[4]);
assert.equal(out.unknown[0].species,'Murkrow');
assert.equal(run({...args,maxProposals:3},()=>rejected).status,'EXHAUSTED_BATCH');
assert.throws(()=>run(args,()=>rows(Array(6).fill('Snorlax'))),/safety cap/);
assert.throws(()=>run(args,()=>[]),/partial party/);
assert.throws(()=>run(args,()=>{throw Error('native failed')}),/native failed/);
let calls=0;
const multiple=run({...args,required:['murkrow']},seed=>{calls++;return seed%2
  ? accepted : rows(['Snorlax','Snorlax','Snorlax','Snorlax','Snorlax','Snorlax']);});
assert.deepEqual(multiple.partySeeds,[0,1]);assert.equal(calls,2);
let tick=0;
const cancelled=run({...args,budgetMs:1},()=>accepted,()=>tick++);
assert.equal(cancelled.status,'DEADLINE_CANCELLED');assert.equal(cancelled.completeProposals,0);
assert.equal(canon('Unown-Question'),'unown');assert.equal(canon('Unown-B'),'unown');
"""
        result=subprocess.run([node,'--input-type=module','-e',script.replace('MODULE',json.dumps(module))],
            capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    @requires_showdown()
    def test_native_bridge_exact_original_draw_and_rng_equivalence(self):
        env=LocalShowdownEnv(LocalShowdownConfig(showdown_root=showdown_root()))
        try:
            source=load_gen3_randbat_source_cached(showdown_root())
            s=PaperHiddenTeamSampler(env,set_source=source)
            for required in (frozenset({'snorlax'}),frozenset({'snorlax','clefable'})):
                for batch in (1,8,16):
                    left=random.Random(93);right=random.Random(93)
                    a=s.draw_membership_first((KnownSetTraits('Snorlax'),),left,
                        required=required,check=lambda:None,receipt=receipt())
                    evidence=receipt()
                    b=draw_native_membership(s,(KnownSetTraits('Snorlax'),),right,
                        required=required,check=lambda:None,receipt=evidence,
                        batch_size=batch,deadline_at=None)
                    self.assertEqual(a,b)
                    self.assertEqual(left.getstate(),right.getstate())
            # Hard-root-like multiple required species: no forcing, truncating
            # the original party, or changing the subsequent known-set draws.
            required=frozenset({'shuckle','clefable','marowak'})
            left=random.Random(47);right=random.Random(47)
            original_receipt=receipt();warm_receipt=receipt()
            known=(KnownSetTraits('Shuckle'),)
            a=s.draw_membership_first(known,left,required=required,
                check=lambda:None,receipt=original_receipt)
            b=draw_native_membership(s,known,right,required=required,check=lambda:None,
                receipt=warm_receipt,batch_size=8,deadline_at=None)
            self.assertEqual(a,b)
            self.assertEqual(left.getstate(),right.getstate())
            self.assertEqual(original_receipt['complete_proposals'],warm_receipt['complete_proposals'])
            self.assertEqual(original_receipt['membership_rejections'],warm_receipt['membership_rejections'])
        finally: env.close()
