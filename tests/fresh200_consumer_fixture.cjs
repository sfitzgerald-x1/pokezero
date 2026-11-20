'use strict';
// Cross-language saved-artifact fixture. Every class below is synthetic;
// no Native tree, game, pool, frozen root, chance source or reference is loaded.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {createObserver} = require('../gpu_engine/fresh200_witness_bridge.cjs');
const {ZERO} = require('../gpu_engine/fresh200_witness_core.cjs');
const {TransportWitness, CHANNELS, READERS, methodHash} = require('../gpu_engine/fresh200_witness_transport.cjs');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const wire = value => Buffer.from(JSON.stringify(value)+'\n');
const out = process.argv[2];assert.ok(out);fs.mkdirSync(out,{recursive:true});
const registrationMode=process.argv[3]==='--registration-policy';

class Reader {
  constructor(){this.buffer=[];this.waiters=[];this.ended=false;}
  [Symbol.asyncIterator](){return this;}
  next(){return this.buffer.length ? Promise.resolve({value:this.buffer.shift(),done:false}) :
    this.ended ? Promise.resolve({value:undefined,done:true}) : new Promise((resolve,reject)=>this.waiters.push({resolve,reject}));}
  push(value){if(this.waiters.length)this.waiters.shift().resolve({value,done:false});else this.buffer.push(value);}
  pushEnd(){this.ended=true;for(const waiter of this.waiters.splice(0))waiter.resolve({value:undefined,done:true});}
  pushError(error){for(const waiter of this.waiters.splice(0))waiter.reject(error);}
}
class Source extends Reader {
  _write(items){for(const item of items){if(typeof item==='string')this.push(item);else item.side.emitRequest(item.value);}return true;}
}
class Side {
  constructor(n,source){this.n=n;this.source=source;this.activeRequest=null;
    this.team=Array.from({length:6},(_,i)=>({species:'synthetic'+i}));
    this.pokemon=this.team.map(set=>({set,level:50,happiness:255,pokeball:'pokeball'}));}
  emitRequest(value){this.activeRequest=value;this.source.push('sideupdate\np'+(this.n+1)+'\n|request|'+JSON.stringify(value));return true;}
}
class Battle {
  constructor(source){this.sides=[new Side(0,source),new Side(1,source)];this.players=[];}
  setPlayer(side){this.players.push(side);return side;}
}
async function main(){
  // Declare original function identities before either witness installs hooks.
  const hooks=[['Battle.setPlayer',Battle.prototype.setPlayer],['Side.emitRequest',Side.prototype.emitRequest]]
    .map(([operation,fn])=>({operation,originalSha256:methodHash(fn),kind:'method'})).sort((a,b)=>a.operation.localeCompare(b.operation));
  const hookDigest=hash(wire(hooks));
  // Synthetic source declarations captured before any hooks are installed.
  // Real Native source parsing/admission is a separate gate; these exact class
  // method spans exercise only the policy/closed-command composition.
  const sourceBytes=fs.readFileSync(__filename);
  const catalogFunctions=hooks.map(entry=>{
    const fn=entry.operation==='Battle.setPlayer'?Battle.prototype.setPlayer:Side.prototype.emitRequest;
    const text=Buffer.from(Function.prototype.toString.call(fn)),start=sourceBytes.indexOf(text);assert.ok(start>=0);
    return {...entry,stage:'bootstrap-method',locations:[{source:__filename,start,end:start+text.length,
      declaration:entry.operation,source_form:'method-definition'}]};
  });
  const expectedMethods={};
  for(const [name,fn] of [
    ['source.iterator',Reader.prototype[Symbol.asyncIterator]],['source.next',Reader.prototype.next],
    ['source.write',Source.prototype._write],['source.push',Reader.prototype.push],['source.end',Reader.prototype.pushEnd],
    ['source.error',Reader.prototype.pushError],['stream.push',Reader.prototype.push],['stream.end',Reader.prototype.pushEnd],['stream.error',Reader.prototype.pushError],
  ])expectedMethods[name]=methodHash(fn);
  const sourceHash=hash(Buffer.from('synthetic consumer fixture source declaration; no actual Native tree'));
  const audits=[],nativeReceipts=[],events=[],completions=[],stdout=[],listeners=[];
  const transport=new TransportWitness({session:'consumer-fixture',battleId:'b',sourceRegistryHash:sourceHash,expectedMethods,
    appendEvent:row=>{events.push(wire(row));for(const waiter of [...listeners])if(waiter.predicate()){
      listeners.splice(listeners.indexOf(waiter),1);waiter.resolve();}}});
  const when=predicate=>predicate()?Promise.resolve():new Promise(resolve=>listeners.push({predicate,resolve}));
  let control,command,transportControl;
  const observer=createObserver({Battle,sourceRegistryHash:sourceHash,transport,registrationEvidence:registrationMode,
    loadControl:()=>control,appendAudit:bytes=>audits.push(bytes),appendReceipt:bytes=>nativeReceipts.push(bytes),
    loadTransportControl:()=>({value:transportControl,sha256:hash(wire(transportControl))}),appendTransportReceipt:bytes=>completions.push(bytes)});
  const core=observer.core;
  core.installMethod(Side.prototype,'emitRequest',{operation:'Side.emitRequest',registrationSite:{kind:'bootstrap-method',owner:'Side'},
    enter(side){transport.requestBegin(core.frames.at(-1).callId,'p'+(side.n+1));return {subject:{kind:'Side',n:side.n}};},
    leave(_before,_side,_args,outcome){transport.requestEnd(core.frames.at(-1).callId,outcome.didThrow);return null;}});
  const source=new Source(),streams=Object.fromEntries(CHANNELS.map(n=>[n,new Reader()])),battle=new Battle(source);
  const emit=packet=>{const bytes=wire(packet);transport.stdout(bytes.toString('utf8'));stdout.push(bytes);};
  let fanout;
  const begin=(index,phase)=>{
    command={type:phase,battleId:'b',requestId:'q'+index};
    control={schema:'pokezero.native_witness_control.v1',run:'consumer-fixture',stream:0,reset:0,command:index,phase,
      request:index,nonce:'consumer-fixture/0/0/'+index,chance_nonce:phase==='choices'?'frozen-fixture-chance':null,previous_receipt_hash:core.lastReceiptHash};
    transportControl={schema:'pokezero.native_transport_control.v1',native_nonce:control.nonce,
      stdin_sha256:hash(wire(command)),command,previous_completion_hash:transport.previousCompletion};
    return observer.commandBegin(command,wire(command).toString('utf8').slice(0,-1));
  };
  const specs=[];
  const finish=async(token,index)=>{
    observer.commandEnd(token,{originalThrew:false});await observer.transportComplete();
    assert.equal(core.poisoned,false);assert.equal(transport.poisoned,false);
    const files={stdin:wire(command),native_control:wire(control),transport_control:wire(transportControl),
      native_receipt:nativeReceipts.at(-1),transport_receipt:completions.at(-1)};
    const entry={};for(const [key,bytes] of Object.entries(files)){
      const name=index+'-'+key+'.jsonl';fs.writeFileSync(path.join(out,name),bytes,{flag:'wx'});entry[key]={path:name,sha256:hash(bytes),bytes:bytes.length};
    }specs.push(entry);
  };
  const consume=async name=>{
    const next=await streams[name].next();assert.equal(next.done,false);const token=transport.readerBegin(name,next.value);
    const lines=next.value.split('\n').filter(Boolean);if(lines.length)emit({type:'stream',battleId:'b',stream:name,lines});transport.readerEnd(token,lines);
  };
  const all=async()=>{for(const name of READERS)await consume(name);};
  let token=begin(0,'start');transport.sourceBegin(source);
  fanout=(async()=>{for await(const value of source){const at=value.indexOf('\n'),kind=value.slice(0,at),body=value.slice(at+1);
    if(kind==='update')for(const name of CHANNELS)streams[name].push(body);
    else if(kind==='sideupdate'){const split=body.indexOf('\n');streams[body.slice(0,split)].push(body.slice(split+1));}
  }for(const stream of Object.values(streams))stream.pushEnd();})();
  transport.streamsBegin(streams);battle.setPlayer('p1');battle.setPlayer('p2');
  source._write(['update\n|turn|1','update\n|turn|1','update\n',
    {side:battle.sides[0],value:null},{side:battle.sides[1],value:{side:{id:'p2'},wait:true}},
    {side:battle.sides[0],value:{side:{id:'p1'},active:[{}]}},{side:battle.sides[0],value:{side:{id:'p1'},active:[{}],update:true}}]);
  await when(()=>events.filter(x=>JSON.parse(x).operation==='FANOUT_ACK').length===7);
  for(let i=0;i<3;i++)await all();await consume('p1');await consume('p2');await consume('p1');await consume('p1');
  const ready=transport.scheduled('ready');emit({type:'ready',battleId:'b'});transport.callbackFinished(ready,false);await finish(token,0);
  token=begin(1,'choices');source._write([{side:battle.sides[0],value:{side:{id:'p1'},wait:true}}]);
  await when(()=>events.filter(x=>JSON.parse(x).operation==='FANOUT_ACK').length===8);await consume('p1');await finish(token,1);
  token=begin(2,'snapshot');observer.snapshotBegin('b',battle,'q2');
  const requests=observer.snapshotRequests('b',battle,'q2'),state={turn:1,note:'literal α snapshot'};
  const witness=observer.snapshotEnd('b',battle,state,requests,'q2');
  emit({type:'snapshot',battleId:'b',requestId:'q2',snapshot:{battle:state,boundaryRequests:requests,witness}});await finish(token,2);
  token=begin(3,'end');transport.teardownBegin();source.pushEnd();transport.teardownEnd();await fanout;
  for(const name of READERS){assert.equal((await streams[name].next()).done,true);emit({type:'stream_end',battleId:'b',stream:name});transport.readerEOF(name);}
  const terminal=transport.scheduled('terminal');transport.callbackFinished(terminal,true);await finish(token,3);
  token=begin(4,'close');await finish(token,4);
  for(const [name,parts] of Object.entries({audit:audits,events,stdout}))fs.writeFileSync(path.join(out,name+'.jsonl'),Buffer.concat(parts),{flag:'wx'});
  fs.writeFileSync(path.join(out,'manifest.json'),wire({synthetic_only:true,no_actual_native_game:true,session:'consumer-fixture',battle_id:'b',
    source_registry_hash:sourceHash,expected_methods:expectedMethods,hook_identity_digest:hookDigest,call_operations:hooks.map(x=>x.operation),commands:specs,
    ...(registrationMode?{registration_catalog:{source_pins:{[__filename]:hash(sourceBytes)},functions:catalogFunctions,
      bootstrap_operations:['Battle.setPlayer','Side.emitRequest'],instance_operations:[]}}:{})}));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
