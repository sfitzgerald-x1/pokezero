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

export const witnessCommandBegin = command => observe('commandBegin', [command]);
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
