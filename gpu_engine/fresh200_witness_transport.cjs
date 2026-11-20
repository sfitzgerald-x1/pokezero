'use strict';

// Passive IO ledger. Native causal rows remain in WitnessCore; asynchronous
// fanout/read observations must not borrow that core's active command lifetime.
// The bridge/consumer integration and independent source admission are separate
// gates. A clean ledger is never physical or actual-device qualification.
const crypto=require('node:crypto');
const FN_TEXT=Function.prototype.toString;
const CHANNELS=Object.freeze(['omniscient','spectator','p1','p2','p3','p4']);
const READERS=Object.freeze(['omniscient','p1','p2']);
const ZERO='0'.repeat(64);
const need=(value,message)=>{if(!value)throw Error(message);};
const hash=value=>crypto.createHash('sha256').update(value).digest('hex');
const hex=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
function text(value){
  need(typeof value==='string','LITERAL_UTF8_STRING_REQUIRED');
  const bytes=Buffer.from(value,'utf8');need(bytes.toString('utf8')===value,'UTF8_ROUNDTRIP_DIFFERS');return bytes;
}
function canonical(value){
  if(Array.isArray(value))return '['+value.map(canonical).join(',')+']';
  if(value&&typeof value==='object')return '{'+Object.keys(value).sort().map(k=>JSON.stringify(k)+':'+canonical(value[k])).join(',')+'}';
  const encoded=JSON.stringify(value);need(encoded!==undefined,'LITERAL_LEDGER_VALUE_REQUIRED');return encoded;
}
function data(object,key){
  for(let current=object;current;current=Object.getPrototypeOf(current)){
    const descriptor=Object.getOwnPropertyDescriptor(current,key);
    if(descriptor){need('value'in descriptor,'TRANSPORT_DATA_PROPERTY_REQUIRED');return descriptor.value;}
  }
  throw Error('MISSING_TRANSPORT_PROPERTY');
}
function methodHash(fn){need(typeof fn==='function','TRANSPORT_METHOD_REQUIRED');return hash(Reflect.apply(FN_TEXT,fn,[]));}
function framed(digest,bytes){const length=Buffer.alloc(8);length.writeBigUInt64LE(BigInt(bytes.length));digest.update(length);digest.update(bytes);}
function raw(value){const bytes=text(value);return {utf8_base64:bytes.toString('base64'),bytes:bytes.length,sha256:hash(bytes),split_segments:value.split('\n').length};}
function channel(){return {produced:[],consumed:0,open:null,ended:false,readerEnded:false,
  producedHash:crypto.createHash('sha256'),consumedHash:crypto.createHash('sha256'),prefix:crypto.createHash('sha256'),
  lineCount:0,generation:0,actionable:false};}

class TransportWitness{
  constructor({session,battleId,sourceRegistryHash,expectedMethods,appendEvent,visibilityPolicySha256=null,maxEvents=100000,maxBytes=64*1024**2}){
    need(typeof session==='string'&&session&&typeof battleId==='string'&&battleId&&hex(sourceRegistryHash),'EXACT_TRANSPORT_IDENTITY_REQUIRED');
    need(expectedMethods&&typeof expectedMethods==='object'&&typeof appendEvent==='function','INDEPENDENT_METHOD_ROSTER_AND_EVENT_SINK_REQUIRED');
    need(Number.isSafeInteger(maxEvents)&&maxEvents>0&&Number.isSafeInteger(maxBytes)&&maxBytes>0,'FINITE_TRANSPORT_BOUNDS_REQUIRED');
    need(visibilityPolicySha256===null||hex(visibilityPolicySha256),'EXACT_OWNER_VISIBILITY_POLICY_DIGEST_REQUIRED');
    this.session=session;this.battleId=battleId;this.sourceRegistryHash=sourceRegistryHash;
    this.visibilityPolicySha256=visibilityPolicySha256;
    this.expectedMethods={...expectedMethods};this.appendEvent=appendEvent;this.maxEvents=maxEvents;this.maxBytes=maxBytes;
    this.sequence=0;this.previous=ZERO;this.eventBytes=0;this.poisoned=false;this.reason=null;this.hooks=[];this.waiters=[];
    this.source=null;this.streams=null;this.iteratorSeen=false;this.nextPending=false;this.current=null;
    this.producers=[];this.delivered=0;this.acknowledged=0;this.writeDepth=0;this.teardownDepth=0;this.sourceEnded=false;this.sourceEOF=false;
    this.channels=Object.fromEntries(CHANNELS.map(name=>[name,channel()]));this.requests=[];this.requestStack=[];this.requestIds=new Set();
    this.callbacks=new Map();this.nextCallback=0;this.command=null;this.commandIds=new Set();this.previousCompletion=ZERO;
    this.stdoutHash=crypto.createHash('sha256');this.stdoutCount=0;
  }
  poison(reason){if(!this.poisoned){this.poisoned=true;this.reason=String(reason);}this._checkWaiters();}
  _safe(fn){try{return fn();}catch(error){this.poison(error.message||'TRANSPORT_OBSERVER_FAILED');return undefined;}}
  _event(operation,evidence){
    const row={schema:'pokezero.native_transport_event.v1',session:this.session,battle_id:this.battleId,
      source_epoch:1,command_id:this.command?.id??null,sequence:this.sequence,previous_hash:this.previous,operation,evidence};
    const bytes=text(canonical(row));need(this.sequence<this.maxEvents&&this.eventBytes+bytes.length<=this.maxBytes,'TRANSPORT_EVIDENCE_BOUND');
    this.appendEvent({...row,hash:hash(bytes)});this.sequence++;this.eventBytes+=bytes.length;this.previous=hash(bytes);
  }
  _install(object,key,name,before,after,failed){
    const original=data(object,key);need(methodHash(original)===this.expectedMethods[name],'TRANSPORT_METHOD_SOURCE_DIFFERS:'+name);
    const witness=this;
    function observed(...args){
      const state=witness._safe(()=>before?.(this,args));let result;
      try{result=Reflect.apply(original,this,args);}catch(error){witness._safe(()=>failed?.(error,state));throw error;}
      witness._safe(()=>after?.(result,state));witness._checkWaiters();return result;
    }
    Object.defineProperty(object,key,{value:observed,writable:true,configurable:true});
    this.hooks.push({object,key,observed,original,name});
  }
  sourceBegin(source){return this._safe(()=>{
    need(!this.source,'SINGLE_NATIVE_SOURCE_REQUIRED');this.source=source;
    this._install(source,Symbol.asyncIterator,'source.iterator',receiver=>{
      need(receiver===source&&!this.iteratorSeen,'SINGLE_NATIVE_FANOUT_ITERATOR_REQUIRED');this.iteratorSeen=true;
    },result=>need(result===source,'ORIGINAL_SOURCE_ITERATOR_IDENTITY_DIFFERS'));
    this._install(source,'_write','source.write',(receiver,args)=>{
      need(receiver===source&&this.command&&!this.command.ended,'NATIVE_WRITE_OUTSIDE_COMMAND');this.writeDepth++;
      if(!this.visibilityPolicySha256||this.command.phase!=='start')return null;
      const ordinal=this.command.visibilityWrites+1;
      need((ordinal===1&&!this.command.visibilityApplied)||(ordinal===2&&this.command.visibilityApplied),
        'OWNER_VISIBILITY_NATIVE_WRITE_ORDER_DIFFERS');
      return {ordinal,sha256:hash(text(args[0]))};
    },(_,startWrite)=>{
      this.writeDepth--;
      if(startWrite){
        this.command.visibilityWrites=startWrite.ordinal;
        this._event('START_WRITE_RETURN',startWrite);
      }else this._event('SOURCE_WRITE_RETURN',{});
    },()=>{this.writeDepth--;this.poison('ORIGINAL_NATIVE_WRITE_THROW');});
    this._install(source,'push','source.push',(receiver,args)=>{
      need(receiver===source&&!this.sourceEnded&&(this.writeDepth>0||this.teardownDepth>0),'NATIVE_OUTPUT_OUTSIDE_WRITE_SPAN');
      const value=args[0];const packet=raw(value);const newline=value.indexOf('\n');const type=newline<0?value:value.slice(0,newline);
      need(['update','sideupdate','end'].includes(type),'UNREPRESENTED_NATIVE_PRODUCER_TYPE');
      let side=null,request=null;
      if(type==='sideupdate'){
        const rest=value.slice(newline+1),split=rest.indexOf('\n');need(split>=0,'MALFORMED_SIDEUPDATE');side=rest.slice(0,split);
        need(['p1','p2','p3','p4'].includes(side),'UNKNOWN_NATIVE_ADDRESSED_STREAM');
        const lines=rest.slice(split+1).split('\n');const requests=lines.filter(x=>x.startsWith('|request|'));
        if(requests.length){
          need(requests.length===1&&lines.length===1&&this.requestStack.length,'UNJOINED_NATIVE_REQUEST_EMISSION');
          request=this.requestStack.at(-1);need(request.side===side&&!request.producer,'REQUEST_EMIT_SOURCE_JOIN_DIFFERS');
        }
      }
      const producer={sequence:this.producers.length+1,value,type,side,request,fanout:new Set()};
      if(request)request.producer=producer.sequence;
      this.producers.push(producer);this._event('SOURCE_PUSH',{producer_sequence:producer.sequence,type,side,request_call_id:request?.id??null,raw:packet});
    });
    this._install(source,'next','source.next',receiver=>{
      need(receiver===source&&this.iteratorSeen&&!this.nextPending,'OVERLAPPING_OR_FOREIGN_SOURCE_NEXT');
      if(this.current){
        const expected=this.current.type==='update'?CHANNELS:this.current.type==='sideupdate'?[this.current.side]:[];
        need(expected.length===this.current.fanout.size&&expected.every(x=>this.current.fanout.has(x)),'NATIVE_FANOUT_CARDINALITY_DIFFERS');
        this.acknowledged=this.current.sequence;this._event('FANOUT_ACK',{producer_sequence:this.current.sequence});this.current=null;
      }
      need(!this.sourceEOF,'SOURCE_NEXT_AFTER_EOF');this.nextPending=true;
    },promise=>{
      Reflect.apply(Promise.prototype.then,promise,[result=>{this._safe(()=>{
        this.nextPending=false;need(result&&typeof result.done==='boolean','LITERAL_NATIVE_ITERATOR_RESULT_REQUIRED');
        if(result.done){need(this.sourceEnded&&this.delivered===this.producers.length,'PREMATURE_NATIVE_SOURCE_EOF');this.sourceEOF=true;this._event('SOURCE_EOF',{});}
        else{const producer=this.producers[this.delivered];need(producer&&producer.value===result.value&&!this.current,'PRODUCER_DELIVERY_FIFO_DIFFERS');
          this.delivered++;this.current=producer;this._event('SOURCE_DELIVERED',{producer_sequence:producer.sequence});}
      });this._checkWaiters();},()=>{this.nextPending=false;this.poison('ORIGINAL_NATIVE_SOURCE_REJECTION');}]);
    },()=>{this.nextPending=false;this.poison('ORIGINAL_NATIVE_SOURCE_NEXT_THROW');});
    this._install(source,'pushEnd','source.end',receiver=>{
      need(receiver===source&&this.teardownDepth>0&&!this.sourceEnded,'UNDECLARED_NATIVE_SOURCE_END');this.sourceEnded=true;this._event('SOURCE_PUSH_END',{});
    });
    this._install(source,'pushError','source.error',()=>this.poison('ORIGINAL_NATIVE_SOURCE_ERROR'));
    this._event('SOURCE_REGISTERED',{methods:this.hooks.map(x=>({name:x.name,sha256:methodHash(x.original)}))});
  });}
  streamsBegin(streams){return this._safe(()=>{
    need(this.source&&!this.streams&&Object.keys(streams).sort().join(',')===[...CHANNELS].sort().join(','),'EXACT_SIX_NATIVE_STREAMS_REQUIRED');this.streams=streams;
    for(const name of CHANNELS){const stream=streams[name],state=this.channels[name];
      this._install(stream,'push','stream.push',(receiver,args)=>{
        need(receiver===stream&&!state.ended&&this.current,'OWNER_PUSH_OUTSIDE_ACTUAL_FANOUT');
        const producer=this.current;need(!producer.fanout.has(name)&&(producer.type==='update'||producer.type==='sideupdate'&&producer.side===name),'DUPLICATE_OR_WRONG_NATIVE_FANOUT');
        const order=producer.type==='update'?CHANNELS:[producer.side];
        need(order[producer.fanout.size]===name,'NATIVE_FANOUT_ORDER_DIFFERS');
        producer.fanout.add(name);const packet=raw(args[0]);const chunk={sequence:state.produced.length+1,producer:producer.sequence,value:args[0],request:producer.request};
        state.produced.push(chunk);framed(state.producedHash,text(chunk.value));
        this._event('STREAM_PUSH',{stream:name,stream_sequence:chunk.sequence,producer_sequence:chunk.producer,raw:packet});
      });
      this._install(stream,'pushEnd','stream.end',receiver=>{
        need(receiver===stream&&this.sourceEOF&&!state.ended,'UNJOINED_NATIVE_STREAM_END');state.ended=true;this._event('STREAM_PUSH_END',{stream:name});
      });
      this._install(stream,'pushError','stream.error',()=>this.poison('ORIGINAL_NATIVE_STREAM_ERROR'));
    }
    this._event('STREAMS_REGISTERED',{names:CHANNELS,required_readers:READERS});
  });}
  teardownBegin(){return this._safe(()=>{need(this.command&&!this.command.ended&&!this.teardownDepth,'EXACT_TEARDOWN_SPAN_REQUIRED');this.teardownDepth++;});}
  teardownEnd(){return this._safe(()=>{need(this.teardownDepth===1,'TEARDOWN_SPAN_DIFFERS');this.teardownDepth--;this._checkWaiters();});}
  requestBegin(id,side){return this._safe(()=>{
    const key=this.command?.id+':'+id;
    need(Number.isSafeInteger(id)&&id>=0&&!this.requestIds.has(key)&&['p1','p2'].includes(side)&&this.writeDepth>0,'EXACT_NATIVE_EMITREQUEST_CALL_REQUIRED');
    const entry={id,side,command:this.command.id,producer:null,returned:false,joined:false};this.requestIds.add(key);this.requests.push(entry);this.requestStack.push(entry);this._event('REQUEST_CALL_BEGIN',{call_id:id,side});
  });}
  requestEnd(id,originalThrew=false){return this._safe(()=>{
    const entry=this.requestStack.pop();need(entry&&entry.id===id&&entry.command===this.command?.id&&!originalThrew&&entry.producer,'NATIVE_EMITREQUEST_RETURN_DIFFERS');entry.returned=true;this._event('REQUEST_CALL_END',{call_id:id});this._checkWaiters();
  });}
  readerBegin(name,value){return this._safe(()=>{
    need(READERS.includes(name),'UNREGISTERED_BRIDGE_READER');const state=this.channels[name],chunk=state.produced[state.consumed];
    need(!state.open&&!state.readerEnded&&chunk&&chunk.value===value,'READER_CHUNK_FIFO_DIFFERS');state.open=chunk;
    this._event('READER_BEGIN',{stream:name,stream_sequence:chunk.sequence});return {stream:name,sequence:chunk.sequence};
  });}
  readerEnd(token,lines){return this._safe(()=>{
    need(token&&READERS.includes(token.stream),'EXACT_READER_TOKEN_REQUIRED');const state=this.channels[token.stream],chunk=state.open;
    need(chunk&&chunk.sequence===token.sequence&&Array.isArray(lines)&&lines.every(x=>typeof x==='string'),'READER_COMPLETION_TOKEN_DIFFERS');
    const expected=chunk.value.split('\n').filter(x=>x.length>0);need(JSON.stringify(lines)===JSON.stringify(expected),'ORIGINAL_EMITTED_PROTOCOL_LINES_DIFFER');
    need(expected.length?Number.isSafeInteger(chunk.stdoutOrdinal):chunk.stdoutOrdinal===undefined,'ORIGINAL_STDOUT_STREAM_PACKET_REQUIRED');
    for(const line of lines){
      state.prefix.update(text(line+'\n'));state.lineCount++;
      if(['p1','p2'].includes(token.stream)&&line.startsWith('|request|')){
        const request=chunk.request;need(request&&request.side===token.stream&&request.returned&&!request.joined&&request.producer===chunk.producer,'REQUEST_READER_JOIN_DIFFERS');
        const value=JSON.parse(line.slice('|request|'.length));need(value===null||value&&typeof value==='object'&&!Array.isArray(value)&&value.side?.id===token.stream,'LITERAL_OWNER_REQUEST_IDENTITY_DIFFERS');
        need(value===null||!('wait'in value)||typeof value.wait==='boolean','LITERAL_WAIT_BIT_REQUIRED');
        state.generation++;state.actionable=!!(value&&!value.wait&&(value.active?.length||value.forceSwitch?.some(Boolean)));request.joined=true;
        this._event('REQUEST_CONSUMED',{owner:token.stream,generation:state.generation,emit_call_id:request.id,emit_command_id:request.command,producer_sequence:chunk.producer,
          stream_sequence:chunk.sequence,line_sha256:hash(text(line)),classification:value===null?'null':value.wait?'wait':state.actionable?'actionable':'inactive'});
      }
      if(line==='|tie'||line.startsWith('|win|'))state.actionable=false;
    }
    framed(state.consumedHash,text(chunk.value));state.consumed++;state.open=null;this._event('READER_END',{stream:token.stream,stream_sequence:chunk.sequence,protocol_lines:lines.length});this._checkWaiters();
  });}
  readerEOF(name){return this._safe(()=>{
    need(READERS.includes(name),'UNREGISTERED_BRIDGE_READER');const state=this.channels[name];need(state.ended&&!state.readerEnded&&!state.open&&state.consumed===state.produced.length&&Number.isSafeInteger(state.eofStdoutOrdinal),'READER_EOF_DIFFERS');state.readerEnded=true;this._event('READER_EOF',{stream:name});this._checkWaiters();
  });}
  scheduled(kind){return this._safe(()=>{need(this.command&&!this.command.completed&&['ready','terminal'].includes(kind),'KNOWN_ORIGINAL_CONTROL_CALLBACK_REQUIRED');const id=this.nextCallback++;this.callbacks.set(id,kind);this._event('CONTROL_SCHEDULED',{id,kind});return id;});}
  callbackFinished(id,discarded=false){return this._safe(()=>{need(this.callbacks.has(id)&&typeof discarded==='boolean','ORIGINAL_CONTROL_CALLBACK_COMPLETION_DIFFERS');this.callbacks.delete(id);this._event('CONTROL_FINISHED',{id,discarded});this._checkWaiters();});}
  stdout(wire){return this._safe(()=>{
    need(this.command&&!this.command.completed,'STDOUT_OUTSIDE_CLOSED_COMMAND');const bytes=text(wire);
    need(wire.endsWith('\n')&&!/[\r\n]/.test(wire.slice(0,-1)),'EXACT_STDOUT_JSON_LINE_REQUIRED');
    const packet=JSON.parse(wire);need(packet&&typeof packet==='object'&&!Array.isArray(packet),'LITERAL_STDOUT_PACKET_REQUIRED');
    need(packet.battleId===undefined||packet.battleId===this.battleId,'STDOUT_BATTLE_IDENTITY_DIFFERS');
    if(packet.type==='stream'){
      need(READERS.includes(packet.stream),'ORIGINAL_STDOUT_READER_REQUIRED');const chunk=this.channels[packet.stream].open;
      need(chunk&&!chunk.stdoutOrdinal&&packet.lines?.length>0&&JSON.stringify(packet.lines)===JSON.stringify(chunk.value.split('\n').filter(x=>x.length>0)),'ORIGINAL_STDOUT_STREAM_JOIN_DIFFERS');
      chunk.stdoutOrdinal=this.stdoutCount+1;
    }else if(packet.type==='stream_end'){
      need(READERS.includes(packet.stream),'ORIGINAL_STDOUT_READER_REQUIRED');const state=this.channels[packet.stream];
      need(state.ended&&!state.readerEnded&&!state.eofStdoutOrdinal,'ORIGINAL_STDOUT_EOF_JOIN_DIFFERS');state.eofStdoutOrdinal=this.stdoutCount+1;
    }
    framed(this.stdoutHash,bytes);this.stdoutCount++;this._event('STDOUT_WRITE',{ordinal:this.stdoutCount,wire:raw(wire)});
  });}
  commandBegin({id,phase,receivedLine,stdinSha256,controlHash}){return this._safe(()=>{
    need(typeof id==='string'&&id&&!this.commandIds.has(id)&&(!this.command||this.command.completed),'UNIQUE_NONOVERLAPPING_TRANSPORT_COMMAND_REQUIRED');
    need(['start','choices','snapshot','end','close'].includes(phase)&&hex(stdinSha256)&&hex(controlHash)&&typeof receivedLine==='string'&&!/[\r\n]/.test(receivedLine),'EXACT_CLOSED_RECEIVED_COMMAND_REQUIRED');
    need(hash(text(receivedLine+'\n'))===stdinSha256,'INDEPENDENT_STDIN_BYTES_DIFFER');
    this.command={id,phase,stdinSha256,controlHash,firstEvent:this.sequence,ended:false,completed:false,
      visibilityWrites:0,visibilityApplied:false};
    this.commandIds.add(id);this._event('COMMAND_RECEIVED',{phase,received_line:receivedLine,stdin_sha256:stdinSha256,control_hash:controlHash});
  });}
  visibilityApplied(evidence){
    need(!this.poisoned&&this.visibilityPolicySha256&&evidence.policy_sha256===this.visibilityPolicySha256
      &&this.command?.phase==='start'&&this.command.visibilityWrites===1&&!this.command.visibilityApplied
      &&!this.writeDepth&&!this.requests.length,'OWNER_VISIBILITY_TRANSITION_OUTSIDE_PRE_PLAYER_WINDOW');
    this._event('START_VISIBILITY_APPLIED',evidence);
    this.command.visibilityApplied=true;
  }
  commandEnd({originalThrew,nativeReceiptHash,snapshotHash=null}){return this._safe(()=>{
    need(this.command&&!this.command.ended&&typeof originalThrew==='boolean'&&hex(nativeReceiptHash)&&(snapshotHash===null||hex(snapshotHash)),'EXACT_NATIVE_COMMAND_OUTCOME_REQUIRED');
    need(!this.visibilityPolicySha256||this.command.phase!=='start'
      ||this.command.visibilityApplied&&this.command.visibilityWrites===2,'INCOMPLETE_OWNER_VISIBILITY_NATIVE_START');
    this.command.ended=true;this.command.nativeReceiptHash=nativeReceiptHash;this.command.snapshotHash=snapshotHash;
    this._event('COMMAND_RETURN',{original_threw:originalThrew,native_receipt_hash:nativeReceiptHash,snapshot_hash:snapshotHash});
    if(originalThrew)this.poison('ORIGINAL_BRIDGE_COMMAND_THROW');this._checkWaiters();
  });}
  _ready(eof){
    for(const hook of this.hooks)need(data(hook.object,hook.key)===hook.observed&&methodHash(hook.original)===this.expectedMethods[hook.name],'TRANSPORT_HOOK_IDENTITY_DRIFT');
    return this.command?.ended&&this.source&&this.streams&&!this.writeDepth&&!this.teardownDepth&&!this.requestStack.length
      &&this.acknowledged===this.producers.length&&!this.current&&!this.callbacks.size&&this.requests.every(x=>x.returned&&x.joined)
      &&READERS.every(name=>!this.channels[name].open&&this.channels[name].consumed===this.channels[name].produced.length)
      &&(!eof||this.sourceEOF&&CHANNELS.every(name=>this.channels[name].ended)&&READERS.every(name=>this.channels[name].readerEnded));
  }
  _completion(eof){
    const streams=Object.fromEntries(CHANNELS.map(name=>{const s=this.channels[name];return [name,{produced:s.produced.length,consumed:s.consumed,
      produced_hash:s.producedHash.copy().digest('hex'),consumed_hash:s.consumedHash.copy().digest('hex'),protocol_lines:s.lineCount,
      prefix_hash:s.prefix.copy().digest('hex'),request_generation:s.generation,actionable:s.actionable,ended:s.ended,reader_ended:s.readerEnded}];}));
    const result={schema:'pokezero.native_transport_completion.v1',session:this.session,battle_id:this.battleId,source_epoch:1,
      command_id:this.command.id,phase:this.command.phase,source_registry_hash:this.sourceRegistryHash,stdin_sha256:this.command.stdinSha256,
      control_hash:this.command.controlHash,native_receipt_hash:this.command.nativeReceiptHash,snapshot_hash:this.command.snapshotHash,
      previous_completion_hash:this.previousCompletion,event_range:[this.command.firstEvent,this.sequence],event_head:this.previous,
      producer_frontier:this.producers.length,fanout_ack_frontier:this.acknowledged,streams,stdout_frontier:this.stdoutCount,
      stdout_hash:this.stdoutHash.copy().digest('hex'),source_eof:this.sourceEOF,require_eof:eof,poisoned:false,qualification:false};
    result.hash=hash(text(canonical(result)));this.previousCompletion=result.hash;this.command.completed=true;this.command.receipt=result;return result;
  }
  _checkWaiters(){
    const pending=this.waiters;this.waiters=[];
    for(const waiter of pending){
      if(this.poisoned){waiter.reject(Error('NO_CLEAN_TRANSPORT_COMPLETION:'+this.reason));continue;}
      let ready=false;try{ready=this._ready(waiter.eof);}catch(error){this.poisoned=true;this.reason=error.message;waiter.reject(error);continue;}
      if(ready)waiter.resolve(this.command.completed?this.command.receipt:this._completion(waiter.eof));else this.waiters.push(waiter);
    }
  }
  fence({requireEOF}={}){return new Promise((resolve,reject)=>{
    if(!this.command){reject(Error('TRANSPORT_COMMAND_REQUIRED'));return;}
    const eof=['end','close'].includes(this.command.phase);
    if(requireEOF!==undefined&&requireEOF!==eof){reject(Error('SOURCE_CLOSURE_PHASE_DIFFERS'));return;}
    this.waiters.push({resolve,reject,eof});this._checkWaiters();
  });}
}

module.exports={TransportWitness,CHANNELS,READERS,methodHash,canonical};
