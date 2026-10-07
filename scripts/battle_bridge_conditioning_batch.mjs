// Transport-only public-history prefilter. Unrecognized chronology is not a
// rejection: stop the batch and let the authoritative Python parser decide.
export function retryPublicSuffix(lines, initialTurn) {
  const result = [];
  let turn = initialTurn;
  for (const raw of lines) {
    // Python str.strip also recognizes these control/NEL characters, unlike
    // JavaScript trim. Never reject a trial using a different normalization.
    if (typeof raw !== 'string' || raw.trim() !== raw ||
        /^[\u001c-\u001f\u0085]|[\u001c-\u001f\u0085]$/.test(raw)) return null;
    if (!raw || raw === '|' || raw.startsWith('>')) continue;
    const parts = raw.split('|');
    const kind = parts[1] || '';
    if (kind === 't:' || (kind === 'request' && parts.length >= 3)) continue;
    if (raw === "|message|The battle's RNG was reset.") continue;
    if (kind === 'upkeep' && raw !== '|upkeep') return null;
    if (kind === 'turn' && parts.length >= 3) {
      if (!/^\|turn\|[1-9][0-9]*$/.test(raw)) return null;
      const next = Number(parts[2]);
      if (!Number.isSafeInteger(next) || (turn !== 0 && next !== turn + 1)) return null;
      turn = next;
    }
    result.push(raw);
  }
  return result;
}

export function samePublicSuffix(actual, expected) {
  return actual !== null && actual.length === expected.length &&
    actual.every((line, index) => line === expected[index]);
}
