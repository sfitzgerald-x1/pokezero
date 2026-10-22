// Launch through a tiny importer in an owned temporary directory. The stock bridge
// and immutable Showdown source are imported, not edited or materialized.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const at=process.argv.indexOf('--showdown-root');
if(at<0)throw Error('explicit pinned Showdown root required');
const root=process.argv[at+1],dir=path.dirname(process.argv[1]);
const control=path.join(dir,'chance.json'),audit=path.join(dir,'chance-audit.jsonl'),initial=path.join(dir,'initializing.json');
const {Battle}=require(path.join(root,'dist/sim/battle.js'));
const {PRNG}=require(path.join(root,'dist/sim/prng.js'));
require('./chance_hooks.cjs').install(Battle,PRNG,
  ()=>fs.existsSync(control)?JSON.parse(fs.readFileSync(control,'utf8')):null,
  row=>fs.appendFileSync(audit,JSON.stringify(row)+'\n'),
  ()=>fs.existsSync(initial)?JSON.parse(fs.readFileSync(initial,'utf8')):null,
  path.join(root,'dist/sim/pokemon.js'));
await import('../scripts/battle_bridge.mjs');
