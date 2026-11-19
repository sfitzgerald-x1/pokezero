// Optional passive observation port. Ordinary bridge users have no observer.
let observer = null;
let ownerVisibility = null;

export function setNativeWitness(value) {
  if (observer !== null || !value || ['commandBegin','commandEnd','snapshotBegin','snapshotEnd','poison']
    .some(name => typeof value[name] !== 'function')) throw Error('EXACT_NATIVE_WITNESS_PORT_REQUIRED');
  observer = value;
}

export function setOwnerVisibility(value) {
  if (ownerVisibility !== null || !value || observer?.transport?.visibilityPolicySha256 !== value.sha256) {
    throw Error('OWNER_VISIBILITY_REQUIRES_BOUND_NATIVE_TRANSPORT');
  }
  ownerVisibility = value;
}

export function requireFreshOwnerVisibilityStart(battleId, battleCount) {
  if (ownerVisibility === null) return false;
  const transport = observer.transport;
  if (battleCount !== 0 || transport.poisoned || transport.command?.phase !== 'start'
      || battleId !== ownerVisibility.policy.battle_id
      || transport.command.stdinSha256 !== ownerVisibility.policy.start_wire_sha256) {
    throw Error('EXACT_FRESH_OWNER_VISIBILITY_START_REQUIRED');
  }
  return true;
}

export async function writeOwnerVisibilityStart(battleStream, writer, start, players, context) {
  if (ownerVisibility === null) throw Error('OWNER_VISIBILITY_NOT_ADMITTED');
  const transport = observer.transport;
  await writer.write(start);
  if (transport.poisoned || transport.command?.visibilityWrites !== 1) {
    throw Error('UNVERIFIED_PRE_PLAYER_NATIVE_START_WRITE');
  }
  const evidence = ownerVisibility.apply(battleStream.battle, {
    ...context, session: transport.session, stdinSha256: transport.command.stdinSha256,
  });
  transport.visibilityApplied(evidence);
  await writer.write(players);
  ownerVisibility.verifyBoundary(battleStream.battle);
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
