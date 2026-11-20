'use strict';
// Synthetic IO fixtures only. Actual Native fanout/bridge and root replay are
// separate integration gates; no simulator, chance draw or policy runs here.
const test=require('node:test');
const assert=require('node:assert/strict');
const crypto=require('node:crypto');
const {TransportWitness,CHANNELS,READERS,methodHash}=require('../gpu_engine/fresh200_witness_transport.cjs');
const hash=value=>crypto.createHash('sha256').update(value).digest('hex');

class Reader{
  constructor(){this.buf=[];this.waiters=[];this.atEOF=false;this.errors=[];this.pushCalls=0;this.nextCalls=0;}
  push(value){this.pushCalls++;if(this.atEOF)return;if(this.waiters.length)this.waiters.shift().resolve({value,done:false});else this.buf.push(value);}
  pushEnd(){this.atEOF=true;for(const waiter of this.waiters.splice(0))waiter.resolve({value:undefined,done:true});}
  pushError(error){this.errors.push(error);for(const waiter of this.waiters.splice(0))waiter.reject(error);}
  [Symbol.asyncIterator](){return this;}
  next(){
    this.nextCalls++;
    const promise=this.errors.length?Promise.reject(this.errors[0]):this.buf.length?Promise.resolve({value:this.buf.shift(),done:false}):
      this.atEOF?Promise.resolve({value:undefined,done:true}):new Promise((resolve,reject)=>this.waiters.push({resolve,reject}));
    this.lastNextPromise=promise;return promise;
  }
}
class Source extends Reader{
  constructor(){super();this.writeCalls=0;this.writeResult={original:true};this.writeError=null;}
  _write(values){
    this.writeCalls++;this.lastWriteReceiver=this;this.lastWriteArgs=values;
    if(this.writeError)throw this.writeError;
    for(const value of values){
      if(typeof value==='string')this.push(value);
      else{this.requestBegin(value.id,value.side);this.push(value.chunk);this.requestEnd(value.id);}
    }
    return this.writeResult;
  }
}
function roster(){return {
  'source.iterator':methodHash(Reader.prototype[Symbol.asyncIterator]),'source.next':methodHash(Reader.prototype.next),
  'source.write':methodHash(Source.prototype._write),'source.push':methodHash(Reader.prototype.push),
  'source.end':methodHash(Reader.prototype.pushEnd),'source.error':methodHash(Reader.prototype.pushError),
  'stream.push':methodHash(Reader.prototype.push),'stream.end':methodHash(Reader.prototype.pushEnd),'stream.error':methodHash(Reader.prototype.pushError),
};}
function begin(w,id='c0',phase='start'){
  const receivedLine=JSON.stringify({type:phase,battleId:'b',requestId:id});
  w.commandBegin({id,phase,receivedLine,stdinSha256:hash(receivedLine+'\n'),controlHash:'1'.repeat(64)});
}
function finish(w,originalThrew=false){w.commandEnd({originalThrew,nativeReceiptHash:'2'.repeat(64)});}
function fixture({skip=null,duplicate=null,sinkOperation=null,buffered=[]}={}){
  const rows=[],listeners=[];let caughtResolve;const caught=new Promise(resolve=>caughtResolve=resolve);
  const w=new TransportWitness({session:'synthetic',battleId:'b',sourceRegistryHash:'3'.repeat(64),expectedMethods:roster(),appendEvent:row=>{
    if(row.operation===sinkOperation)throw Error('SYNTHETIC_EVENT_SINK_FAILED');
    rows.push(row);for(const listener of [...listeners])if(listener.predicate(rows)){listeners.splice(listeners.indexOf(listener),1);listener.resolve(rows);}
  }});
  begin(w);const source=new Source();source.requestBegin=(id,side)=>w.requestBegin(id,side);source.requestEnd=id=>w.requestEnd(id);
  w.sourceBegin(source);if(buffered.length)source._write(buffered);
  const streams=Object.fromEntries(CHANNELS.map(name=>[name,new Reader()]));
  // The fixture keeps the inspected source's synchronous fanout before its
  // next existing iterator invocation; no drain timer or extra read is added.
  const fanout=(async()=>{
    try{
      for await(const chunk of source){
        const split=chunk.indexOf('\n'),kind=chunk.slice(0,split),body=chunk.slice(split+1);
        if(kind==='update')for(const name of CHANNELS){if(name===skip)continue;streams[name].push(body);if(name===duplicate)streams[name].push(body);}
        else if(kind==='sideupdate'){const at=body.indexOf('\n');streams[body.slice(0,at)].push(body.slice(at+1));}
      }
      for(const stream of Object.values(streams))stream.pushEnd();
    }catch(error){for(const stream of Object.values(streams))stream.pushError(error);caughtResolve(error);}
  })();
  w.streamsBegin(streams);
  const when=predicate=>predicate(rows)?Promise.resolve(rows):new Promise(resolve=>listeners.push({predicate,resolve}));
  return {w,source,streams,rows,fanout,caught,when,ack:count=>when(rows=>rows.filter(x=>x.operation==='FANOUT_ACK').length>=count)};
}
async function consume(f,name,{delay=false,emit=true}={}){
  const result=await f.streams[name].next();assert.equal(result.done,false);
  const token=f.w.readerBegin(name,result.value),lines=result.value.split('\n').filter(x=>x.length>0);
  if(emit&&lines.length)f.w.stdout(JSON.stringify({type:'stream',battleId:'b',stream:name,lines})+'\n');
  if(!delay)f.w.readerEnd(token,lines);
  return {token,lines,value:result.value};
}
async function consumeAll(f){for(const name of READERS)await consume(f,name);}
const request=(id,value)=>({id,side:'p1',chunk:'sideupdate\np1\n|request|'+JSON.stringify(value)});

test('buffered startup joins all six actual fanout pushes and required readers before drain',async()=>{
  const f=fixture({buffered:['update\n|turn|1']});await f.ack(1);finish(f.w);
  let settled=false;const proof=f.w.fence();proof.then(()=>settled=true);await Promise.resolve();assert.equal(settled,false);
  const delayed=await consume(f,'p1',{delay:true});await consume(f,'omniscient');await consume(f,'p2');
  await Promise.resolve();assert.equal(settled,false);f.w.readerEnd(delayed.token,delayed.lines);
  const receipt=await proof;assert.equal(receipt.producer_frontier,1);assert.equal(receipt.fanout_ack_frontier,1);
  for(const name of CHANNELS)assert.equal(receipt.streams[name].produced,1);
  for(const name of READERS)assert.equal(receipt.streams[name].produced_hash,receipt.streams[name].consumed_hash);
  assert.equal(receipt.qualification,false);assert.equal(receipt.source_eof,false);
});

test('duplicate bytes and empty chunks retain separate FIFO ordinals and raw boundaries',async()=>{
  const f=fixture();f.source._write(['update\n|turn|1','update\n|turn|1','update\n']);await f.ack(3);
  for(let index=0;index<3;index++)await consumeAll(f);finish(f.w);const receipt=await f.w.fence();
  assert.equal(receipt.streams.p1.produced,3);assert.equal(receipt.streams.p1.protocol_lines,2);
  assert.equal(receipt.stdout_frontier,6);assert.equal(f.rows.filter(x=>x.operation==='STREAM_PUSH').length,18);
  assert.equal(f.rows.filter(x=>x.operation==='READER_END'&&x.evidence.protocol_lines===0).length,3);
});

test('every null wait and identical update request joins its actual call and advances generation',async()=>{
  const f=fixture();const wait={side:{id:'p1'},wait:true};const active={side:{id:'p1'},active:[{}],update:true};
  f.source._write([request(0,null),request(1,wait),request(2,wait),request(3,active),request(4,active)]);await f.ack(5);
  for(let i=0;i<5;i++)await consume(f,'p1');finish(f.w);const first=await f.w.fence();
  assert.equal(first.streams.p1.request_generation,5);assert.equal(first.streams.p1.actionable,true);
  const joined=f.rows.filter(x=>x.operation==='REQUEST_CONSUMED');assert.deepEqual(joined.map(x=>x.evidence.emit_call_id),[0,1,2,3,4]);
  assert.deepEqual(joined.map(x=>x.evidence.classification),['null','wait','wait','actionable','actionable']);
  begin(f.w,'c1','choices');f.source._write([request(0,wait)]);await f.ack(6);await consume(f,'p1');finish(f.w);
  const second=await f.w.fence();assert.equal(second.streams.p1.request_generation,6);assert.equal(second.streams.p1.actionable,false);
  assert.equal(second.previous_completion_hash,first.hash);
});

test('original control callback completion is required and terminal wire invalidates actionability',async()=>{
  const f=fixture();f.source._write([request(0,{side:{id:'p1'},active:[{}]})]);await f.ack(1);await consume(f,'p1');
  const callback=f.w.scheduled('ready');finish(f.w);let settled=false;const proof=f.w.fence();proof.then(()=>settled=true);
  await Promise.resolve();assert.equal(settled,false);f.w.callbackFinished(callback,false);await proof;
  begin(f.w,'c1','choices');f.source._write(['update\n|win|literal']);await f.ack(2);await consumeAll(f);finish(f.w);
  const receipt=await f.w.fence();assert.equal(receipt.streams.p1.actionable,false);assert.equal(receipt.source_eof,false);
});

test('explicit end requires source EOF all six stream ends three reader EOF packets',async()=>{
  const f=fixture();finish(f.w);await f.w.fence();begin(f.w,'c1','end');
  f.w.teardownBegin();f.source.pushEnd();f.w.teardownEnd();finish(f.w);
  await f.when(rows=>rows.filter(x=>x.operation==='STREAM_PUSH_END').length===6);
  let settled=false;const proof=f.w.fence();proof.then(()=>settled=true);await Promise.resolve();assert.equal(settled,false);
  for(const name of READERS){assert.equal((await f.streams[name].next()).done,true);f.w.stdout(JSON.stringify({type:'stream_end',battleId:'b',stream:name})+'\n');f.w.readerEOF(name);}
  const receipt=await proof;assert.equal(receipt.require_eof,true);assert.equal(receipt.source_eof,true);await f.fanout;
  await assert.rejects(f.w.fence({requireEOF:false}),/SOURCE_CLOSURE_PHASE_DIFFERS/);
});

test('original write receiver arguments return object and thrown object remain unchanged',async()=>{
  const f=fixture();const input=['update\n|turn|1'];const value=f.source._write(input);
  assert.equal(value,f.source.writeResult);assert.equal(f.source.lastWriteReceiver,f.source);assert.equal(f.source.lastWriteArgs,input);assert.equal(f.source.writeCalls,1);
  await f.ack(1);await consumeAll(f);finish(f.w);await f.w.fence();
  begin(f.w,'c1','choices');const error=Error('ORIGINAL_WRITE');f.source.writeError=error;
  assert.throws(()=>f.source._write([]),actual=>actual===error);assert.equal(f.source.writeCalls,2);finish(f.w,true);
  await assert.rejects(f.w.fence(),/NO_CLEAN_TRANSPORT_COMPLETION/);
});

test('overlapping next poisons evidence while preserving the original Promise identity',async()=>{
  const f=fixture();const promise=f.source.next();assert.equal(promise,f.source.lastNextPromise);assert.equal(f.source.nextCalls,2);
  await assert.rejects(f.w.fence(),/OVERLAPPING_OR_FOREIGN_SOURCE_NEXT/);
});

test('source rejection reaches the original fanout catch with the same error object',async()=>{
  const f=fixture();const error=Error('ORIGINAL_SOURCE_ERROR');f.source.pushError(error,true);
  assert.equal(await f.caught,error);for(const stream of Object.values(f.streams))assert.equal(stream.errors[0],error);
  await assert.rejects(f.w.fence(),/NO_CLEAN_TRANSPORT_COMPLETION/);
});

test('missing duplicate and out-of-interval owner fanout cannot produce completion',async()=>{
  for(const config of [{skip:'p2'},{duplicate:'p1'}]){
    const f=fixture(config);f.source._write(['update\n|turn|1']);await f.when(rows=>rows.some(x=>x.operation==='SOURCE_DELIVERED'));
    await assert.rejects(f.w.fence(),/NO_CLEAN_TRANSPORT_COMPLETION/);
  }
  const f=fixture();f.streams.p1.push('|turn|1');assert.equal(f.streams.p1.pushCalls,1);
  await assert.rejects(f.w.fence(),/OWNER_PUSH_OUTSIDE_ACTUAL_FANOUT/);
});

test('unjoined request and missing actual stdout stream packet fail closed',async()=>{
  const unjoined=fixture();unjoined.source._write(['sideupdate\np1\n|request|null']);
  await assert.rejects(unjoined.w.fence(),/UNJOINED_NATIVE_REQUEST_EMISSION/);
  const f=fixture();f.source._write(['update\n|turn|1']);await f.ack(1);await consume(f,'p1',{emit:false});
  await assert.rejects(f.w.fence(),/ORIGINAL_STDOUT_STREAM_PACKET_REQUIRED/);
});

test('observer sink errors and invalid UTF8 never suppress original writes',async()=>{
  const f=fixture({sinkOperation:'SOURCE_PUSH'});const result=f.source._write(['update\n|turn|1']);
  assert.equal(result,f.source.writeResult);assert.equal(f.source.writeCalls,1);await f.ack(1);
  for(const stream of Object.values(f.streams))assert.equal(stream.pushCalls,1);
  await assert.rejects(f.w.fence(),/SYNTHETIC_EVENT_SINK_FAILED/);
  const bad=fixture();const input='update\n\ud800';assert.equal(bad.source._write([input]),bad.source.writeResult);
  assert.equal(bad.source.writeCalls,1);await assert.rejects(bad.w.fence(),/UTF8_ROUNDTRIP_DIFFERS/);
});

test('independent method pins hook drift and late command throw defeat clean-looking snapshots',async()=>{
  const wrong=new TransportWitness({session:'synthetic',battleId:'b',sourceRegistryHash:'3'.repeat(64),expectedMethods:{...roster(),'source.iterator':'0'.repeat(64)},appendEvent:()=>{}});
  begin(wrong);wrong.sourceBegin(new Source());await assert.rejects(wrong.fence(),/TRANSPORT_METHOD_SOURCE_DIFFERS/);
  const drift=fixture();drift.source.push=Reader.prototype.push;finish(drift.w);await assert.rejects(drift.w.fence(),/TRANSPORT_HOOK_IDENTITY_DRIFT/);
  const late=fixture();late.w.commandEnd({originalThrew:true,nativeReceiptHash:'2'.repeat(64),snapshotHash:'4'.repeat(64)});
  await assert.rejects(late.w.fence(),/ORIGINAL_BRIDGE_COMMAND_THROW/);
});
