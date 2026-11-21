'use strict';
// Synthetic cross-language wire fixture only. No Native/engine/game import.
const fs = require('node:fs');
const path = require('node:path');
const {WitnessCore,hash} = require('../gpu_engine/fresh200_witness_core.cjs');
const directory = process.argv[2];
const audit = path.join(directory,'audit.jsonl'), receipts = path.join(directory,'receipts.jsonl');
const source = '3'.repeat(64), commands = [];
const core = new WitnessCore({appendAudit:bytes=>fs.appendFileSync(audit,bytes),sourceRegistryHash:source});
const holder = {move(value) {return value;}};
core.installMethod(holder,'move',{operation:'fixture.move',enter:(_subject,args)=>({move:args[0]})});
const snapshot = {battle:{turn:9,weight:1e-7,side:'\u03b1',optional:undefined},boundaryRequests:{p1:{active:[{moves:[{id:'tackle'}]}]},p2:{wait:true}}};
let packet;
for (const [command,phase] of ['start','choices','snapshot'].entries()) {
  const control = {schema:'pokezero.native_witness_control.v1',run:'wire-fixture',stream:7,reset:0,command,phase,
    request:command,nonce:'wire-fixture/7/0/'+command,chance_nonce:phase==='choices'?'7:0':null,previous_receipt_hash:core.lastReceiptHash};
  const token = core.begin(control,{battleId:'fixture',phase});holder.move('tackle');
  if (phase==='snapshot') {
    const state = Buffer.from(JSON.stringify(snapshot.battle)), requests = Buffer.from(JSON.stringify(snapshot.boundaryRequests));
    packet = {state_wire_base64:state.toString('base64'),state_sha256:hash(state),
      requests_wire_base64:requests.toString('base64'),requests_sha256:hash(requests)};
  }
  const receipt = core.end(token,{boundary:phase==='snapshot'?packet:null});
  fs.appendFileSync(receipts,JSON.stringify(receipt)+'\n');
  commands.push({control,battle_id:'fixture',hook_identity_digest:receipt.hookIdentityDigest});
}
fs.writeFileSync(path.join(directory,'manifest.json'),JSON.stringify({source_registry_hash:source,
  expected_commands:commands,call_operations:['fixture.move'],snapshot,packet,qualification:false})+'\n',{flag:'wx'});
