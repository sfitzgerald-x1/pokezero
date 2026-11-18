// Explicit same-prefix validation bridge. No Native PRNG draws are permitted.
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
  path.join(root,'dist/sim/pokemon.js'),{strictNoRng:true});
const witnessControl=path.join(dir,'witness-control.json');
if(fs.existsSync(witnessControl)){
  const {setNativeWitness}=await import('../scripts/battle_bridge_witness_port.mjs');
  const registry={nativeRoot:root,node:process.version,
    helpers:Object.fromEntries(['fresh200_witness_core.cjs','fresh200_witness_bridge.cjs','chance_hooks.cjs','fresh200_bridge.mjs']
      .map(name=>[name,require('node:crypto').createHash('sha256').update(fs.readFileSync(new URL(name,import.meta.url))).digest('hex')])),
    bridge:require('node:crypto').createHash('sha256').update(fs.readFileSync(new URL('../scripts/battle_bridge.mjs',import.meta.url))).digest('hex'),
    port:require('node:crypto').createHash('sha256').update(fs.readFileSync(new URL('../scripts/battle_bridge_witness_port.mjs',import.meta.url))).digest('hex'),
    setPlayer:require('node:crypto').createHash('sha256').update(Function.prototype.toString.call(Battle.prototype.setPlayer)).digest('hex')};
  const sourceRegistryHash=require('node:crypto').createHash('sha256').update(JSON.stringify(registry)+'\n').digest('hex');
  fs.writeFileSync(path.join(dir,'witness-source-registry.json'),JSON.stringify({registry,sourceRegistryHash})+'\n',{flag:'wx'});
  setNativeWitness(require('./fresh200_witness_bridge.cjs').createObserver({Battle,sourceRegistryHash,
    loadControl:()=>JSON.parse(fs.readFileSync(witnessControl,'utf8')),
    appendAudit:bytes=>fs.appendFileSync(path.join(dir,'witness-audit.jsonl'),bytes),
    appendReceipt:bytes=>fs.appendFileSync(path.join(dir,'witness-receipts.jsonl'),bytes)}));
}
await import('../scripts/battle_bridge.mjs');
