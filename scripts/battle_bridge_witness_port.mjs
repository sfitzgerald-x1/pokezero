// Optional passive observation port. Ordinary bridge users have no observer.
let observer = null;

export function setNativeWitness(value) {
  if (observer !== null || !value || ['commandBegin','commandEnd','snapshotBegin','snapshotEnd','poison']
    .some(name => typeof value[name] !== 'function')) throw Error('EXACT_NATIVE_WITNESS_PORT_REQUIRED');
  observer = value;
}

function observe(name, args) {
  if (observer === null) return undefined;
  try { return Reflect.apply(observer[name], observer, args); }
  catch {
    // Diagnostics cannot suppress an original operation or replace its error.
    // A missing/poisoned receipt makes the reference unsuccessful.
    try { observer.poison('BRIDGE_OBSERVER_FAILED'); } catch {}
    return undefined;
  }
}

export const witnessCommandBegin = (command, receivedLine) => observe('commandBegin', [command, receivedLine]);
export const witnessCommandEnd = (token, outcome) => observe('commandEnd', [token, outcome]);
export const witnessSnapshotBegin = (battleId, battle, requestId) => observe('snapshotBegin', [battleId, battle, requestId]);
export const witnessSnapshotRequests = (battleId,battle,requestId) => observe('snapshotRequests',[battleId,battle,requestId]);
export function snapshotRequestsForCommand(battleId,battle,requestId,{passive,getRequests}){
  if(!passive)return getRequests();
  const requests=witnessSnapshotRequests(battleId,battle,requestId);
  if(requests===undefined)throw Error('EXPLICIT_PASSIVE_NATIVE_WITNESS_REQUIRED');
  return requests;
}
export const witnessSnapshotEnd = (battleId, battle, state, requests, requestId) =>
  observe('snapshotEnd', [battleId, battle, state, requests, requestId]);

export const nativeTransportEnabled = () => observer !== null && !!observer.transport;
function transportObserve(name, args, battleId) {
  if (!nativeTransportEnabled()) return undefined;
  try {
    if (battleId !== undefined && battleId !== observer.transport.battleId) throw Error('TRANSPORT_BATTLE_IDENTITY_DIFFERS');
    return Reflect.apply(observer.transport[name], observer.transport, args);
  } catch {
    try { observer.transport.poison('BRIDGE_TRANSPORT_OBSERVER_FAILED'); } catch {}
    try { observer.poison('BRIDGE_TRANSPORT_OBSERVER_FAILED'); } catch {}
    return undefined;
  }
}
export const witnessTransportSource = (battleId, source) => transportObserve('sourceBegin', [source], battleId);
export const witnessTransportStreams = (battleId, streams) => transportObserve('streamsBegin', [streams], battleId);
export const witnessTransportReaderBegin = (battleId, stream, chunk) => transportObserve('readerBegin', [stream, chunk], battleId);
export const witnessTransportReaderEnd = (battleId, token, lines) => transportObserve('readerEnd', [token, lines], battleId);
export const witnessTransportReaderEOF = (battleId, stream) => transportObserve('readerEOF', [stream], battleId);
export const witnessTransportTeardownBegin = battleId => transportObserve('teardownBegin', [], battleId);
export const witnessTransportTeardownEnd = battleId => transportObserve('teardownEnd', [], battleId);
export const witnessTransportScheduled = (battleId, kind) => transportObserve('scheduled', [kind], battleId);
export const witnessTransportCallbackFinished = (battleId, token, discarded) => transportObserve('callbackFinished', [token, discarded], battleId);
export const witnessTransportStdout = wire => transportObserve('stdout', [wire]);
export const witnessTransportAbort = reason => transportObserve('poison', [reason]);
export const witnessTransportComplete = () => nativeTransportEnabled() ? observe('transportComplete', []) : undefined;
