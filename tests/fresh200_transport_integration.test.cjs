'use strict';
// Changed-composition adapter guards only. Native runtime/root replay remains
// an explicit separate gate; all classes and control declarations below are synthetic.
const test=require('node:test');
const assert=require('node:assert/strict');
const crypto=require('node:crypto');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {createObserver}=require('../gpu_engine/fresh200_witness_bridge.cjs');
const {WitnessCore,ZERO}=require('../gpu_engine/fresh200_witness_core.cjs');
const {createRecorder,METHODS}=require('../gpu_engine/fresh200_witness_native.cjs');
const {TransportWitness,CHANNELS,methodHash}=require('../gpu_engine/fresh200_witness_transport.cjs');
const hash=value=>crypto.createHash('sha256').update(value).digest('hex');
const control=(n,phase,parent=ZERO)=>({schema:'pokezero.native_witness_control.v1',run:'transport-fixture',stream:0,reset:0,
  command:n,phase,request:n,nonce:'transport-fixture/0/0/'+n,chance_nonce:null,previous_receipt_hash:parent});
function fakeTransport(){return {battleId:'b',previousCompletion:ZERO,poisoned:false,calls:[],
  poison(code){this.poisoned=true;this.calls.push(['poison',code]);},
  commandBegin(value){this.calls.push(['begin',value]);},commandEnd(value){this.calls.push(['end',value]);},
  fence(){this.calls.push(['fence']);return Promise.resolve({schema:'synthetic-only',hash:'8'.repeat(64),qualification:false});}};}
function fixture(transport=fakeTransport(),{receiptError=null}={}){
  class Battle{
    constructor(){this.players=[];this.sides=[0,1].map(n=>{
      const team=Array.from({length:6},(_,i)=>({species:'synthetic'+i}));
      return {n,team,pokemon:team.map(set=>({set,level:50,happiness:255,pokeball:'pokeball'})),activeRequest:{side:{id:'p'+(n+1)},active:[{}]}};
    });}
    setPlayer(side){this.players.push(side);return side;}
  }
  let command={type:'start',battleId:'b'},receivedLine=JSON.stringify(command),native=control(0,'start'),reads=0,override=null;
  const receipts=[],completed=[],rows=[];
  const observer=createObserver({Battle,sourceRegistryHash:'7'.repeat(64),transport,
    loadControl:()=>native,appendAudit:bytes=>rows.push(bytes),appendReceipt:bytes=>receipts.push(bytes),
    loadTransportControl:()=>{reads++;const value={schema:'pokezero.native_transport_control.v1',native_nonce:native.nonce,
      previous_completion_hash:transport.previousCompletion,command,stdin_sha256:hash(receivedLine+'\n')};
      if(override)override(value);return {value,sha256:hash(JSON.stringify(value)+'\n')};},
    appendTransportReceipt:bytes=>{if(receiptError)throw receiptError;completed.push(bytes);}});
  const battle=new Battle();
  return {observer,transport,battle,receipts,completed,rows,get reads(){return reads;},
    begin(){return observer.commandBegin(command,receivedLine);},
    advance(n,phase){command={type:phase,battleId:'b',requestId:'q'};receivedLine=JSON.stringify(command);native=control(n,phase,observer.core.lastReceiptHash);},
    override(fn){override=fn;}};
}

test('transport ports remain no-ops for an ordinary causal observer and forward original arguments',async()=>{
  const port=await import(pathToFileURL(path.resolve('scripts/battle_bridge_witness_port.mjs')).href+'?transport-off');
  const calls=[],token={original:true};port.setNativeWitness({commandBegin(...args){calls.push(args);return token;},commandEnd(){},snapshotBegin(){},snapshotEnd(){},poison(){assert.fail('ordinary observer must not be poisoned');}});
  const command={type:'start'};assert.equal(port.witnessCommandBegin(command,'wire'),token);assert.equal(calls[0][0],command);assert.equal(calls[0][1],'wire');
  assert.equal(port.nativeTransportEnabled(),false);assert.equal(port.witnessTransportSource('b',{}),undefined);
  assert.equal(port.witnessTransportStdout('ignored'),undefined);assert.equal(port.witnessTransportComplete(),undefined);
});

test('enabled ports preserve chunk identity and reject a foreign battle without invoking its observer',async()=>{
  const port=await import(pathToFileURL(path.resolve('scripts/battle_bridge_witness_port.mjs')).href+'?transport-on');
  const calls=[],poisons=[],token={original:true},transport={battleId:'b',readerBegin(...args){calls.push(args);return token;},poison(code){poisons.push(code);}};
  port.setNativeWitness({transport,commandBegin(){},commandEnd(){},snapshotBegin(){},snapshotEnd(){},poison(code){poisons.push(code);}});
  const chunk={toString(){assert.fail('port must not coerce original chunks');}};
  assert.equal(port.witnessTransportReaderBegin('b','p1',chunk),token);assert.equal(calls[0][1],chunk);
  assert.equal(port.witnessTransportReaderBegin('foreign','p1',chunk),undefined);assert.equal(calls.length,1);assert.equal(poisons.length,2);
});

test('command outcome and separately drained receipt are joined and appended once',async()=>{
  const f=fixture(),token=f.begin();assert.equal(f.battle.setPlayer('p1'),'p1');assert.equal(f.battle.setPlayer('p2'),'p2');
  const native=f.observer.commandEnd(token,{originalThrew:false});assert.equal(native.poisoned,false);
  const end=f.transport.calls.find(x=>x[0]==='end')[1];assert.equal(end.nativeReceiptHash,hash(f.receipts[0]));
  let release;f.transport.fence=()=>new Promise(resolve=>release=resolve);
  const first=f.observer.transportComplete();assert.equal(f.observer.transportComplete(),first);assert.equal(f.completed.length,0);
  release({schema:'synthetic-only',qualification:false});await first;assert.equal(f.completed.length,1);assert.equal(f.reads,1);
  assert.deepEqual(f.battle.players,['p1','p2']);
});

test('early clean snapshot cannot override a later original command throw',async()=>{
  const f=fixture(),start=f.begin();f.battle.setPlayer('p1');f.battle.setPlayer('p2');f.observer.commandEnd(start,{originalThrew:false});await f.observer.transportComplete();
  f.advance(1,'snapshot');const token=f.begin();f.observer.snapshotBegin('b',f.battle,'q');
  const requests=f.observer.snapshotRequests('b',f.battle,'q');const snapshot=f.observer.snapshotEnd('b',f.battle,{turn:1},requests,'q');
  assert.equal(snapshot.receipt.poisoned,false);assert.equal(f.observer.core.active,null);
  f.observer.commandEnd(token,{originalThrew:true});assert.equal(f.transport.poisoned,true);
  await assert.rejects(f.observer.transportComplete(),/NO_CLEAN_TRANSPORT_COMPLETION/);assert.equal(f.completed.length,1);
});

test('independent command declaration and exact received stdin bytes are required',async()=>{
  const mismatch=fixture();mismatch.override(value=>{value.command={type:'choices'};});
  assert.throws(()=>mismatch.begin(),/INDEPENDENT_TRANSPORT_COMMAND_BINDING_DIFFERS/);assert.equal(mismatch.transport.calls.length,0);
  const transport=new TransportWitness({session:'synthetic',battleId:'b',sourceRegistryHash:'7'.repeat(64),expectedMethods:{},appendEvent:()=>{}});
  const f=fixture(transport);const command={type:'start',battleId:'b'};
  const token=f.observer.commandBegin(command,JSON.stringify(command)+' ');assert.equal(transport.poisoned,true);
  f.observer.commandEnd(token,{originalThrew:false});await assert.rejects(f.observer.transportComplete(),/NO_CLEAN_TRANSPORT_COMPLETION/);
});

test('transport receipt sink failure preserves original work and legacy mode reads no transport control',async()=>{
  const error=Error('SYNTHETIC_RECEIPT_IO'),f=fixture(fakeTransport(),{receiptError:error}),token=f.begin();
  f.battle.setPlayer('p1');f.battle.setPlayer('p2');f.observer.commandEnd(token,{originalThrew:false});
  await assert.rejects(f.observer.transportComplete(),actual=>actual===error);assert.deepEqual(f.battle.players,['p1','p2']);assert.equal(f.transport.poisoned,true);
  const legacy=fixture(null),old=legacy.begin();legacy.battle.setPlayer('p1');legacy.battle.setPlayer('p2');
  assert.equal(legacy.observer.commandEnd(old,{originalThrew:false}).poisoned,false);assert.equal(legacy.observer.transportComplete(),undefined);assert.equal(legacy.reads,0);
});

test('one Native emitRequest seam supplies the actual causal call ID and preserves result and throw identity',()=>{
  const classes={Battle:class{},Pokemon:class{addVolatile(){return true;}},Side:class{
    constructor(){this.n=0;this.calls=0;this.returnValue={original:true};}
    emitRequest(value){this.calls++;if(this.failure)throw this.failure;this.activeRequest=value;return this.returnValue;}
  },BattleActions:class{},BattleQueue:class{},Field:class{}};
  for(const [kind,names] of Object.entries(METHODS))for(const name of names)if(!Object.hasOwn(classes[kind].prototype,name))
    Object.defineProperty(classes[kind].prototype,name,{value:function(){return null;},writable:true,configurable:true});
  const rows=[],core=new WitnessCore({sourceRegistryHash:'7'.repeat(64),appendAudit:bytes=>rows.push(JSON.parse(bytes))});
  const observed=[],transport={battleId:'b',requestBegin(id,side){observed.push(['begin',id,side,core.frames.at(-1).operation]);},requestEnd(id,threw){observed.push(['end',id,threw]);}};
  const recorder=createRecorder(core,classes,{transport}),side=new classes.Side();core.begin(control(0,'start'),{battleId:'b',phase:'start'});
  const packet={side:{id:'p1'},wait:true};assert.equal(side.emitRequest(packet),side.returnValue);assert.equal(side.activeRequest,packet);assert.equal(side.calls,1);
  const call=rows.find(row=>row.operation==='CALL_ENTER'&&row.frame.operation==='Side.emitRequest').frame.callId;
  assert.deepEqual(observed,[['begin',call,'p1','Side.emitRequest'],['end',call,false]]);
  const error=Error('ORIGINAL_REQUEST_THROW');side.failure=error;assert.throws(()=>side.emitRequest(packet),actual=>actual===error);assert.equal(side.calls,2);
  assert.equal(observed.at(-1)[2],true);assert.equal(recorder.manifest().guardedEncorePointerPreserved,true);
});

test('a permuted six-channel fanout is refused even when cardinality would eventually match',async()=>{
  class IO{
    [Symbol.asyncIterator](){return this;}
    _write(value){this.push(value);}
    push(value){this.value=value;this.pushes=(this.pushes||0)+1;}
    pushEnd(){}
    pushError(){}
    next(){return Promise.resolve({value:this.value,done:false});}
  }
  const expected={};
  for(const [name,key] of [['source.iterator',Symbol.asyncIterator],['source.write','_write'],['source.push','push'],['source.end','pushEnd'],['source.error','pushError'],['source.next','next'],['stream.push','push'],['stream.end','pushEnd'],['stream.error','pushError']])
    expected[name]=methodHash(IO.prototype[key]);
  const source=new IO(),streams=Object.fromEntries(CHANNELS.map(name=>[name,new IO()]));
  const witness=new TransportWitness({session:'synthetic',battleId:'b',sourceRegistryHash:'7'.repeat(64),expectedMethods:expected,appendEvent:()=>{}});
  const receivedLine=JSON.stringify({type:'start',battleId:'b'});
  witness.commandBegin({id:'c0',phase:'start',receivedLine,stdinSha256:hash(receivedLine+'\n'),controlHash:'1'.repeat(64)});
  witness.sourceBegin(source);source[Symbol.asyncIterator]();witness.streamsBegin(streams);
  source._write('update\n|turn|1');await source.next();streams.p2.push('|turn|1');
  assert.equal(streams.p2.pushes,1);await assert.rejects(witness.fence(),/NATIVE_FANOUT_ORDER_DIFFERS/);
});
