"""Warm immutable setup must preserve complete teams and original seed laws."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from _showdown_root import requires_showdown, showdown_root


@requires_showdown()
class Gen3PartyGeneratorTests(unittest.TestCase):
    def test_fresh_and_reused_generators_match_complete_teams_and_reset_flags(self):
        node=shutil.which('node')
        if node is None:self.skipTest('node unavailable')
        module=(Path(__file__).resolve().parents[1]/'scripts/battle_bridge_gen3_party.mjs').as_uri()
        sim=showdown_root()/'dist/sim/index.js'
        script=r"""
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import crypto from 'node:crypto';
import {createIsolatedGen3PartyGenerator as create} from MODULE;
const require=createRequire(import.meta.url);
const {Teams}=require(SIM);
const parts = seed => {const d=crypto.createHash('sha256').update(seed+':scenario-team').digest();
  return [0,2,4,6].map(i=>d.readUInt16BE(i));};
let constructions=0, shared;
const api={getGenerator(...args){constructions++;shared=Teams.getGenerator(...args);return shared;}};
const warm=create(api);
let wobbuffet=0,ditto=0;
const seeds=[0,1,17,101,0xffffffff,...Array.from({length:512},(_,i)=>(i*0x9e3779b1)>>>0)];
for (const seed of [...seeds,...seeds.slice(0,32).reverse()]) {
  const p=parts(seed);
  const fresh=Teams.getGenerator('gen3randombattle',p);
  const original=fresh.getTeam({seed:p});
  const reused=warm(p);
  assert.deepEqual(reused,original,'complete team differs at seed '+seed);
  assert.equal(shared.prng.getSeed(),fresh.prng.getSeed(),'post-team native RNG differs');
  wobbuffet+=original.some(s=>s.species==='Wobbuffet');
  ditto+=original.some(s=>s.species==='Ditto');
}
assert.equal(constructions,1);
assert(wobbuffet>0 && ditto>0,'battle-wide flags were not exercised');
assert.throws(()=>shared.getPokemonPool('Water',[{species:'Tauros'}],false,Object.keys(shared.randomSets)),
  /pool contract drift/);
assert.throws(()=>warm([1,2,3,true]),/party seed/);
assert.throws(()=>create({getGenerator(){return {constructor:{name:'Wrong'}};}})([1,2,3,4]),
  /Unsupported/);
console.log(JSON.stringify({complete_teams_checked:seeds.length+32,wobbuffet,ditto,constructions}));
"""
        result=subprocess.run([node,'--input-type=module','-e',
            script.replace('MODULE',json.dumps(module)).replace('SIM',json.dumps(str(sim)))],
            capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        evidence=json.loads(result.stdout)
        self.assertEqual(evidence['constructions'],1)
        self.assertGreater(evidence['complete_teams_checked'],500)


if __name__=='__main__':unittest.main()
