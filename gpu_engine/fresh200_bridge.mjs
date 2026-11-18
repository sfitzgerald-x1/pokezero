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
  const nativeMode=fs.existsSync(path.join(dir,'witness-native.json'));
  const transportPath=path.join(dir,'witness-transport.json'),transportMode=fs.existsSync(transportPath);
  if(transportMode&&!nativeMode)throw Error('TRANSPORT_REQUIRES_ACTUAL_NATIVE_REQUEST_OBSERVER');
  const transportBytes=transportMode?fs.readFileSync(transportPath):null;
  const transportDeclaration=transportMode?JSON.parse(transportBytes):null;
  const digest=value=>require('node:crypto').createHash('sha256').update(value).digest('hex');
  let transportFunctions=null;
  if(transportMode){
    if(Object.keys(transportDeclaration).sort().join(',')!=='battle_id,completion_fd,expected_methods,schema,session'
        ||transportDeclaration.schema!=='pokezero.native_transport_declaration.v1'
        ||!Number.isInteger(transportDeclaration.completion_fd)||transportDeclaration.completion_fd<3||transportDeclaration.completion_fd>255)
      throw Error('EXPLICIT_TRANSPORT_DECLARATION_REQUIRED');
    const streamModule=require(path.join(root,'dist/sim/battle-stream.js'));
    const streamLibrary=require(path.join(root,'dist/lib/streams.js'));
    const methodText=Function.prototype.toString;
    const method=(target,key)=>{
      for(let owner=target;owner;owner=Object.getPrototypeOf(owner)){
        const descriptor=Object.getOwnPropertyDescriptor(owner,key);
        if(descriptor){if(typeof descriptor.value!=='function')throw Error('LITERAL_STREAM_METHOD_REQUIRED');return digest(Reflect.apply(methodText,descriptor.value,[]));}
      }
      throw Error('PINNED_STREAM_METHOD_MISSING');
    };
    const original=streamModule.BattleStream.prototype,reader=streamLibrary.ObjectReadStream.prototype;
    const actual={'source.iterator':method(original,Symbol.asyncIterator),'source.next':method(original,'next'),'source.write':method(original,'_write'),
      'source.push':method(original,'push'),'source.end':method(original,'pushEnd'),'source.error':method(original,'pushError'),
      'stream.push':method(reader,'push'),'stream.end':method(reader,'pushEnd'),'stream.error':method(reader,'pushError')};
    if(JSON.stringify(Object.keys(actual).sort())!==JSON.stringify(Object.keys(transportDeclaration.expected_methods).sort())
        ||Object.keys(actual).some(key=>actual[key]!==transportDeclaration.expected_methods[key]))throw Error('INDEPENDENT_STREAM_METHOD_ROSTER_DIFFERS');
    transportFunctions={methods:actual,getPlayerStreams:digest(Reflect.apply(methodText,streamModule.getPlayerStreams,[])),
      battleStreamFile:digest(fs.readFileSync(path.join(root,'dist/sim/battle-stream.js'))),streamLibraryFile:digest(fs.readFileSync(path.join(root,'dist/lib/streams.js'))),
      declaration:digest(transportBytes)};
  }
  const registry={nativeRoot:root,node:process.version,
    nativeMode,transportMode,transportFunctions,
    helpers:Object.fromEntries(['fresh200_witness_core.cjs','fresh200_witness_bridge.cjs','chance_hooks.cjs','fresh200_bridge.mjs',...(nativeMode?['fresh200_witness_native.cjs']:[]),...(transportMode?['fresh200_witness_transport.cjs']:[])]
      .map(name=>[name,require('node:crypto').createHash('sha256').update(fs.readFileSync(new URL(name,import.meta.url))).digest('hex')])),
    bridge:require('node:crypto').createHash('sha256').update(fs.readFileSync(new URL('../scripts/battle_bridge.mjs',import.meta.url))).digest('hex'),
    port:require('node:crypto').createHash('sha256').update(fs.readFileSync(new URL('../scripts/battle_bridge_witness_port.mjs',import.meta.url))).digest('hex'),
    setPlayer:require('node:crypto').createHash('sha256').update(Function.prototype.toString.call(Battle.prototype.setPlayer)).digest('hex')};
  const sourceRegistryHash=require('node:crypto').createHash('sha256').update(JSON.stringify(registry)+'\n').digest('hex');
  fs.writeFileSync(path.join(dir,'witness-source-registry.json'),JSON.stringify({registry,sourceRegistryHash})+'\n',{flag:'wx'});
  const transport=transportMode?new (require('./fresh200_witness_transport.cjs').TransportWitness)({
    session:transportDeclaration.session,battleId:transportDeclaration.battle_id,sourceRegistryHash,expectedMethods:transportDeclaration.expected_methods,
    appendEvent:row=>fs.appendFileSync(path.join(dir,'witness-transport-events.jsonl'),JSON.stringify(row)+'\n')}):null;
  const nativeFactory=nativeMode?core=>require('./fresh200_witness_native.cjs').createRecorder(core,{Battle,
    Pokemon:require(path.join(root,'dist/sim/pokemon.js')).Pokemon,Side:require(path.join(root,'dist/sim/side.js')).Side,
    BattleActions:require(path.join(root,'dist/sim/battle-actions.js')).BattleActions,BattleQueue:require(path.join(root,'dist/sim/battle-queue.js')).BattleQueue,
    Field:require(path.join(root,'dist/sim/field.js')).Field},{transport}):null;
  setNativeWitness(require('./fresh200_witness_bridge.cjs').createObserver({Battle,sourceRegistryHash,nativeFactory,transport,
    loadControl:()=>JSON.parse(fs.readFileSync(witnessControl,'utf8')),
    appendAudit:bytes=>fs.appendFileSync(path.join(dir,'witness-audit.jsonl'),bytes),
    appendReceipt:bytes=>fs.appendFileSync(path.join(dir,'witness-receipts.jsonl'),bytes),
    loadTransportControl:()=>{const bytes=fs.readFileSync(path.join(dir,'witness-transport-control.json'));return {value:JSON.parse(bytes),sha256:digest(bytes)};},
    appendTransportReceipt:bytes=>{
      // Receipt notification follows durable journal writes. It never enters
      // a Native stream or the original bridge stdout protocol.
      for(const name of ['witness-audit.jsonl','witness-receipts.jsonl','witness-transport-events.jsonl']){
        const fd=fs.openSync(path.join(dir,name),'r');try{fs.fsyncSync(fd);}finally{fs.closeSync(fd);}
      }
      const fd=fs.openSync(path.join(dir,'witness-transport-receipts.jsonl'),'a');
      try{fs.writeFileSync(fd,bytes);fs.fsyncSync(fd);}finally{fs.closeSync(fd);}
      let offset=0;
      while(offset<bytes.length){const wrote=fs.writeSync(transportDeclaration.completion_fd,bytes,offset,bytes.length-offset);if(wrote<=0)throw Error('TRANSPORT_RECEIPT_FD_CLOSED');offset+=wrote;}
    }}));
}
await import('../scripts/battle_bridge.mjs');
