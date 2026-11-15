"""Parity of the device strike likelihood (gpu_engine/device_set_posterior.py) with the host set_posterior.

Input: the set_posterior parity corpus (JSONL or JSONL.gz rows, one per Strike of every EncounterPosterior at every
decision; each row reaching the damage core carries the core's exact inputs and, per hard variant reached, the attacker
participant, its 16 rolls, ambiguity and count). Every unique strike runs once through the device kernel with the
device's own variant tables; every row is then checked against it.

Checks (each mismatch is classified and sampled; nothing is skipped silently):
- tables: each recorded attacker participant (level, stats, ability, item) equals the device table's variant;
- per variant reached by the core: the 16 rolls, the count and the ambiguity flag equal set_posterior's;
- admitted rows: the device's verdict over the row's hard support admits, and its survivor bits equal Strike.counts > 0;
- type-immune variants (set_posterior d411aa15, corpus v2: empty rolls, count 0, the strike admitted): the device's
  immune flag and count 0; on a v1 corpus the core-unsupported:hiddenpower rows are checked for the immune flag on the
  stopping variant and counted as the rule change;
- rows refused inside the core: every earlier hard variant is clean on the device and the stopping variant carries
  the refusal's flag;
- rows refused before the core: the device's static move class names the same refusal when set_posterior's reason
  is a move class, and no earlier one otherwise (single-variant and every later reason: none);
- per decision and attacker species: narrow_pin over the admitted strikes in line order gives set_posterior's
  support (hard & every admitted strike's counts > 0; the corpus has no off-model drops);
- divergence of the device's strike-time verdict: a strike's verdict at its first decision against every later
  decision's (the hard support only shrinks), by direction and reason;
- the in-tree guard: admitted rows with an engine-unmodeled modifier (type-boost item, Marvel Scale, Plus/Minus)
  active on a hard variant, by cause;
- known departure: Guts boosting a special move's SpA in tier2 (Showdown's Guts is onModifyAtk only), counted.

usage: python scripts/set_posterior_device_parity.py --corpus DIR_OR_FILES --metadata META.json --out OUT_DIR
       [--showdown-root ROOT] [--shards 0,1,...] [--batch 4096] [--limit-rows N]
Runs on CPU (JAX_PLATFORMS=cpu); the kernel is the only JAX code.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

STATUS = {None: 0, '': 0, 'par': 1, 'brn': 2, 'psn': 3, 'tox': 4, 'slp': 5, 'frz': 6}
WEATHER = {'': 0, 'none': 0, 'raindance': 1, 'sunnyday': 2, 'sandstorm': 3, 'hail': 4}
BOOSTS = ('atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion')
STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
EFFECTIVENESS = {None: 0, '': 0, 'supereffective': 1, 'resisted': -1}
CORE_REASONS = ('core-unsupported', 'ambiguous-band', 'effectiveness-disagreement')


def strike_key(row):
    return (row['ordinal'], row['line'], row['attacker_species'])


def core_fingerprint(row):
    """The core's inputs of a row, which every decision's copy of the same strike must repeat."""
    keep = {k: row.get(k) for k in ('move', 'observed', 'ko', 'crit', 'hp_before', 'effectiveness_line', 'weather',
                                     'pursuit_intercept', 'context', 'defender_participant', 'defender_types', 'core')}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


class Mapper:
    """Corpus names -> the kernel's engine-id inputs."""

    def __init__(self, metadata, source_tables):
        from pokezero.dex import normalize_id
        from pokezero import tier2
        self.norm, self.tier2 = normalize_id, tier2
        vocab = metadata['vocabulary']
        self.ids = {k: {normalize_id(n): i for i, n in enumerate(vocab[k]) if i} for k in ('moves', 'types', 'abilities', 'items', 'species')}
        aliases = [i for i, n in enumerate(vocab['moves']) if i and normalize_id(n).startswith('hiddenpower')]
        self.hidden_power = self.ids['moves'].get('hiddenpower', aliases[0] if aliases else 0)
        self.source = source_tables

    def move(self, name):
        name = self.tier2.canonical_move_id(name)
        return self.hidden_power if name.startswith('hiddenpower') else self.ids['moves'].get(name, 0)

    def get(self, kind, name):
        return self.ids[kind].get(self.norm(name or ''), 0)

    def inputs(self, row):
        """(fields dict, problems) for one core row."""
        problems = []
        ctx, defender = row['context'], row['defender_participant']
        species = self.source.species_id(row['attacker_species'])
        if not species:problems.append('unmapped-species')
        move = self.move(row['move'])
        if not move:problems.append('unmapped-move:' + row['move'])
        fraction = float(ctx.get('attacker_hp_fraction', 1.0))
        percent = int(round(fraction * 100))
        if float(percent) / float(100) != fraction:problems.append('attacker-fraction-not-percent')
        types = [self.get('types', t) for t in row['defender_types']][:2]
        types += [0] * (2 - len(types))
        if not all(types[:len(row['defender_types'])]):problems.append('unmapped-defender-type')
        weather = row.get('weather') or ''
        if weather not in WEATHER:problems.append('weather:' + weather)
        boosts = lambda b:[int(b.get(k, 0)) for k in BOOSTS]
        fields = dict(species=species, move=move, crit=bool(row['core']['token']['crit']), weather=WEATHER.get(weather, 0),
            pursuit=bool(row['core']['token'].get('pursuit_intercept', False)), attacker_status=STATUS.get(ctx.get('attacker_status'), 0),
            attacker_boosts=boosts(ctx.get('attacker_boosts') or {}), attacker_percent=percent,
            attacker_flash_fire=bool(ctx.get('attacker_flash_fire')),
            defender_stats=[int(defender['stats'][k]) for k in STATS], defender_boosts=boosts(ctx.get('defender_boosts') or {}),
            defender_types=types, defender_ability=self.get('abilities', defender.get('ability')),
            defender_item=self.get('items', defender.get('item')), defender_species=self.get('species', defender.get('species_key')),
            defender_status=STATUS.get(ctx.get('defender_status'), 0), hp_before=int(row['hp_before']),
            observed=int(row['observed'] if row['observed'] is not None else -1), ko=bool(row['ko']),
            effectiveness=EFFECTIVENESS.get(row.get('effectiveness_line'), 0), precondition=0)
        if defender.get('ability') and not fields['defender_ability']:problems.append('unmapped-defender-ability')
        if defender.get('item') and not fields['defender_item']:problems.append('unmapped-defender-item')
        if ctx.get('attacker_status') not in STATUS or ctx.get('defender_status') not in STATUS:problems.append('status')
        return fields, problems




class Kernel:
    """The device kernel and reduce, jitted once at a fixed batch (rows padded with the first row)."""

    def __init__(self, dsp, tables, batch):
        import jax
        self.dsp, self.tables, self.batch, self.jax = dsp, tables, batch, jax
        self._strike = jax.jit(dsp.strike)
        self._reduce = jax.jit(lambda result, hard, pre:dsp.reduce(result, hard, pre, guard=False))
        self._guard = jax.jit(lambda result, hard, pre:dsp.reduce(result, hard, pre, guard=True))

    def _chunks(self, n):
        for start in range(0, n, self.batch):
            yield start, min(n, start + self.batch)

    def _pad(self, array, n):
        pad = self.batch - n
        return array if not pad else np.concatenate([array, np.repeat(array[:1], pad, axis=0)])

    def strike(self, fields):
        import jax.numpy as j
        dsp = self.dsp;out = collections.defaultdict(list)
        boolean = ('crit', 'pursuit', 'attacker_flash_fire', 'ko')
        for a, b in self._chunks(len(fields)):
            chunk = fields[a:b]
            cols = {name: j.asarray(self._pad(np.asarray([f[name] for f in chunk], bool if name in boolean else np.int32), b - a))
                    for name in dsp.StrikeInputs._fields}
            result = self.jax.device_get(self._strike(self.tables, dsp.StrikeInputs(**cols)))
            for name in dsp.StrikeResult._fields:out[name].append(np.asarray(getattr(result, name))[:b - a])
        return {k: np.concatenate(v) for k, v in out.items()}

    def reduce(self, result, index, hard_words, *, guard):
        """Device verdicts for rows (strike index, hard words [N,2] uint32); precondition 0 (rows reaching the core)."""
        import jax.numpy as j
        dsp = self.dsp;fn = self._guard if guard else self._reduce
        codes, fits = [], []
        for a, b in self._chunks(len(index)):
            at = self._pad(index[a:b], b - a)
            part = dsp.StrikeResult(None, *(j.asarray(result[name][at]) for name in dsp.StrikeResult._fields[1:]))
            reduced = self.jax.device_get(fn(part, j.asarray(self._pad(hard_words[a:b], b - a)), j.zeros(self.batch, j.int32)))
            codes.append(np.asarray(reduced.reason)[:b - a]);fits.append(np.asarray(reduced.fit)[:b - a])
        return np.concatenate(codes), np.concatenate(fits)


def words(bits):
    out = np.zeros(2, np.uint32)
    for i in np.flatnonzero(bits):out[i // 32] |= np.uint32(1 << (i % 32))
    return out


def bits_of(words_):
    return np.asarray([(int(words_[i // 32]) >> (i % 32)) & 1 for i in range(64)], bool)


def lowest_name(dsp, mask):
    return dsp.REASONS[(mask & -mask).bit_length() - 1] if mask else ''


PARSE_CLASSES = ('struggle', 'fixed-damage', 'ohko', 'conditional-power', 'multi-hit')


class Check:
    def __init__(self, samples):
        self.stats, self.samples, self.limit = collections.Counter(), collections.defaultdict(list), samples

    def bad(self, kind, detail):
        self.stats['mismatch:' + kind] += 1
        if len(self.samples[kind]) < self.limit:self.samples[kind].append(detail)


def is_core(row):
    return 'variants' in row or row['admitted']


def check_shard(rows, kernel, mapper, host, check, tables_np):
    """Every check of the module docstring on one corpus shard (strikes never cross shards: the key holds the ordinal)."""
    dsp = kernel.dsp;stats = check.stats
    strikes, keys, fields, problems = {}, [], [], {}
    for r in rows:
        if not is_core(r):continue
        k = strike_key(r);fp = core_fingerprint(r)
        if k in strikes:
            if strikes[k] != fp:check.bad('strike-inputs-differ-across-decisions', dict(key=k))
            continue
        strikes[k] = fp;keys.append(k)
        f, p = mapper.inputs(r);fields.append(f)
        if p:problems[k] = p
    for p in problems.values():
        for x in p:stats['input-problem:' + x] += 1
    result = kernel.strike(fields) if fields else None
    at = {k: i for i, k in enumerate(keys)}
    admitted_rows, guts_strikes = [], set()
    for r in rows:
        stats['rows'] += 1
        reason, admitted, k = r['reason'], r['admitted'], strike_key(r)
        stats['reason:' + (reason or 'ADMITTED')] += 1
        if not is_core(r):
            # Refused before the core, or single-variant: the static move class alone.
            mv = mapper.move(r['move'])
            device = lowest_name(dsp, int(host['move_class'][mv]) if mv else 0)
            want = 'conditional-power' if reason.startswith('conditional-power') else reason
            if want in PARSE_CLASSES:
                # Eruption / Water Spout below full HP and Revenge after a hit are state facts (the kernel's percent rule
                # and the caller's bit); a -hitcount or a second -damage line makes multi-hit a protocol fact.
                ok = device == want or (want == 'conditional-power' and r['move'] in ('eruption', 'waterspout', 'revenge')) \
                    or (want == 'multi-hit' and not device)
            else:
                ok = want == 'called' or not device
            stats['parse-rows'] += 1
            if not ok:check.bad('parse-move-class', dict(key=k, reason=reason, device=device, move=r['move']))
            continue
        stats['core-rows'] += 1
        if k in problems:
            stats['core-rows-with-input-problems'] += 1;continue
        i = at[k];s = fields[i]['species'];entries = r.get('variants', [])
        for e in entries:
            v, a = e['index'], e['attacker']
            stats['variant-entries'] += 1
            if host['level'][s, v] != a['level'] or host['stats'][s, v].tolist() != [a['stats'][x] for x in STATS]:
                check.bad('table-participant', dict(key=k, index=v, corpus=a, device=[int(host['level'][s, v])] + host['stats'][s, v].tolist()))
            if host['ability'][s, v] != mapper.get('abilities', a['ability']) or host['item'][s, v] != mapper.get('items', a['item']):
                check.bad('table-ability-item', dict(key=k, index=v, ability=a['ability'], item=a['item']))
            # tier2 multiplies a statused Guts holder's SpA too (Showdown's Guts is onModifyAtk only; the engine agrees).
            special = (not tables_np['physical'][fields[i]['move']] if not r['move'].startswith('hiddenpower')
                       else bool(tables_np['special_type'][tables_np['hp_type'][s, v]]))
            guts = a['ability'] == 'guts' and bool(r['context'].get('attacker_status')) and special
            if guts:guts_strikes.add(k)
            if e['rolls']:
                rolls = result['rolls'][i, v].tolist()
                if rolls != list(e['rolls']):
                    check.bad('rolls-tier2-guts-spa' if guts else 'rolls', dict(key=k, index=v, move=r['move'], device=rolls, corpus=e['rolls']))
                    continue
                if int(result['counts'][i, v]) != e['count']:
                    check.bad('count', dict(key=k, index=v, device=int(result['counts'][i, v]), corpus=e['count']))
            elif admitted:
                # set_posterior v2 (d411aa15): a type-immune variant fits 0 rolls and the strike stays admitted.
                stats['immune-variant-entries'] += 1
                if not result['immune'][i, v] or int(result['counts'][i, v]) != 0 or e['count'] != 0:
                    check.bad('immune-variant', dict(key=k, index=v, device_immune=bool(result['immune'][i, v]),
                                                     device_count=int(result['counts'][i, v]), corpus_count=e['count']))
            elif not (result['immune'][i, v] or result['unsupported'][i, v]):
                check.bad('no-rolls-but-device-supported', dict(key=k, index=v, reason=reason))
            if bool(result['ambiguous'][i, v]) != bool(e['ambiguous']):
                check.bad('ambiguous', dict(key=k, index=v, device=bool(result['ambiguous'][i, v]), corpus=e['ambiguous']))
        hard = np.zeros(64, bool)
        for e in entries:hard[e['index']] = True
        if admitted:
            if len(entries) != r['hard_size']:check.bad('admitted-entries-vs-hard', dict(key=k, entries=len(entries), hard=r['hard_size']))
            admitted_rows.append((r, i, hard))
            continue
        # Refused inside the core: the earlier hard variants are clean and the stopping one carries the reason's flag.
        stats['core-refused-rows'] += 1
        stop = entries[-1]['index'] if entries else None
        clean = all(not (result['unsupported'][i, e['index']] or result['ambiguous'][i, e['index']] or result['effdis'][i, e['index']])
                    for e in entries[:-1])
        flag = {'core-unsupported:hiddenpower': 'immune', 'ambiguous-band': 'ambiguous', 'effectiveness-disagreement': 'effdis'}.get(
            reason, 'unsupported' if reason.startswith('core-unsupported') else None)
        if flag is None:
            stats['core-refused-other:' + reason] += 1   # a parse-level reason on a row that reached the core: none expected
            ok = False
        else:
            ok = stop is not None and clean and bool(result[flag][i, stop])
        if reason == 'core-unsupported:hiddenpower':stats['v2-rule:immune-hidden-power-rows'] += 1
        if not ok:check.bad('core-refusal' + ('-tier2-guts-spa' if k in guts_strikes else ''), dict(key=k, reason=reason, stop=stop,
            flags={n: [int(result[n][i, e['index']]) for e in entries] for n in ('unsupported', 'ambiguous', 'effdis', 'immune', 'counts')}))
    verdicts = {}
    if admitted_rows:
        # Admitted rows through the device's reduce (no guard; then the guard's extra refusals by cause).
        index = np.asarray([i for _, i, _ in admitted_rows], np.int32)
        hard_words = np.stack([words(h) for _, _, h in admitted_rows])
        codes, fits = kernel.reduce(result, index, hard_words, guard=False)
        guarded, _ = kernel.reduce(result, index, hard_words, guard=True)
        for (r, i, hard), code, fit, g in zip(admitted_rows, codes, fits, guarded):
            k = strike_key(r);stats['admitted-rows'] += 1
            want = np.zeros(64, bool);want[:len(r['strike_counts'])] = np.asarray(r['strike_counts']) > 0
            device = bits_of(fit)
            if code:
                check.bad('admitted-but-device-refuses' + ('-tier2-guts-spa' if k in guts_strikes else ''),
                          dict(key=k, device=dsp.REASONS[int(code)]))
            elif not np.array_equal(device, want):
                check.bad('survivors', dict(key=k, device=np.flatnonzero(device).tolist(), corpus=np.flatnonzero(want).tolist()))
            if g and not code:stats['guard:' + dsp.REASONS[int(g)]] += 1
            verdicts[(k, r['decision_index'])] = (hard, device if not code else None, want)
    # A strike's verdict at its first decision vs every later decision (the hard support only shrinks), by direction.
    first = {}
    for r in sorted(rows, key=lambda r:r['decision_index']):
        k = strike_key(r)
        if k not in first:first[k] = (r['decision_index'], r['admitted'], r['reason'])
        elif first[k][1] != r['admitted']:
            direction = 'first-refused-later-admitted' if not first[k][1] else 'first-admitted-later-refused'
            stats[f'verdict-change:{direction}:{first[k][2] or r["reason"]}'] += 1
    check_supports(rows, verdicts, dsp, check)


def check_supports(rows, verdicts, dsp, check):
    """Per decision and attacker species with a known hard support (an admitted strike lists it): narrow_pin over the
    decision's admitted strikes in line order equals set_posterior's support; and the in-tree device's support, which
    applied each strike once at its first decision (the strike-time hard; the device never re-reads an old strike),
    compared with it: equal, or wider (the device narrowed less), narrower, or a pin conflict (project then falls back
    to the hard support and sets pin_conflict)."""
    import jax.numpy as j
    first = {}
    for r in sorted(rows, key=lambda r:r['decision_index']):
        k = strike_key(r)
        if k not in first:
            first[k] = verdicts.get((k, r['decision_index']), (None, None, None))[1] if r['admitted'] else None
    groups = collections.defaultdict(list)
    for r in rows:groups[(r['ordinal'], r['decision_index'], r['attacker_species'])].append(r)
    for (ordinal, d, species), members in groups.items():
        members.sort(key=lambda r:r['line'])
        known = [verdicts[(strike_key(r), d)] for r in members if r['admitted']]
        if not known:continue
        hard = known[0][0]
        expected = hard.copy()
        for h, _, want in known:
            if not np.array_equal(h, hard):check.bad('decision-hard-differs', dict(key=(ordinal, d, species)))
            expected &= want
        for label, fits in (('root', [verdicts[(strike_key(r), d)][1] for r in members if r['admitted']]),
                            ('strike-time', [first.get(strike_key(r)) for r in members])):
            pin, active, conflicts = j.zeros((1, 2), j.uint32), j.zeros(1, bool), j.zeros(1, j.int32)
            for fit in fits:
                if fit is None:continue
                pin, active, conflicts, _ = dsp.narrow(pin, active, conflicts, j.asarray(words(fit)[None]), j.asarray([True]))
            standing = bits_of(np.asarray(pin)[0]) if bool(active[0]) else np.ones(64, bool)
            device = hard & standing
            conflict = not device.any()
            if conflict:device = hard
            check.stats[f'supports-{label}'] += 1
            if label == 'root':
                if int(conflicts[0]) or conflict or not np.array_equal(device, expected):
                    check.bad('support', dict(key=(ordinal, d, species), device=np.flatnonzero(device).tolist(),
                                              corpus=np.flatnonzero(expected).tolist(), conflicts=int(conflicts[0])))
                continue
            kind = ('pin-conflict' if conflict else 'equal' if np.array_equal(device, expected) else
                    'device-wider' if (expected & ~device).sum() == 0 else 'device-narrower' if (device & ~expected).sum() == 0
                    else 'device-different')
            check.stats['strike-time-support:' + kind] += 1
            if kind != 'equal' and len(check.samples['strike-time-support:' + kind]) < check.limit:
                check.samples['strike-time-support:' + kind].append(dict(key=(ordinal, d, species),
                    device=np.flatnonzero(device).tolist(), corpus=np.flatnonzero(expected).tolist()))


def shard_files(paths, shards):
    files = []
    for path in paths:
        p = Path(path)
        files += sorted(p.glob('strikes-*.jsonl*')) if p.is_dir() else [p]
    if shards is not None:
        files = [f for f in files if int(f.name.split('-')[1].split('.')[0]) in shards]
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', nargs='+', required=True, help='corpus directories (strikes-NN.jsonl[.gz]) or files')
    parser.add_argument('--metadata', required=True, help='the engine metadata JSON (vocabulary, moves, type_chart)')
    parser.add_argument('--showdown-root', default=os.environ.get('POKEZERO_SHOWDOWN_ROOT'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--shards', default=None, help='comma list of shard numbers (default: all)')
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--samples', type=int, default=20)
    args = parser.parse_args(argv)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.randbat import load_gen3_randbat_source_cached
    from gpu_engine import device_set_posterior as dsp, public_belief
    dex = load_showdown_dex_cached(args.showdown_root)
    source = load_gen3_randbat_source_cached(args.showdown_root)
    metadata = json.loads(Path(args.metadata).read_text())
    source_tables = public_belief.compile_source(source)
    tables, unknown = dsp.compile_tables(source, dex, metadata, source_tables=source_tables)
    host = {name: np.asarray(getattr(tables, name)) for name in ('level', 'stats', 'ability', 'item', 'move_class')}
    tables_np = dict(physical=np.asarray(tables.physical), hp_type=np.asarray(tables.hp_type),
                     special_type=np.asarray(tables.variable.special_type))
    constants = {}
    try:
        from pokezero import set_posterior as sp
        constants = {name: getattr(sp, name) == getattr(dsp, name)
                     for name in ('FIXED_DAMAGE_MOVES', 'OHKO_MOVES', 'CONDITIONAL_POWER_MOVES', 'MULTI_HIT_MOVES')}
    except ImportError:
        constants = {'set_posterior': 'not importable'}
    mapper = Mapper(metadata, source_tables)
    kernel = Kernel(dsp, tables, args.batch)
    check = Check(args.samples)
    files = shard_files(args.corpus, None if args.shards is None else {int(x) for x in args.shards.split(',')})
    for n, f in enumerate(files):
        opener = gzip.open if f.suffix == '.gz' else open
        with opener(f, 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        check_shard(rows, kernel, mapper, host, check, tables_np)
        print(f'{f.name}: {len(rows)} rows, {time.time() - t0:.0f}s, mismatches so far '
              f'{sum(v for k, v in check.stats.items() if k.startswith("mismatch:"))}', flush=True)
    mismatches = sum(v for k, v in check.stats.items() if k.startswith('mismatch:'))
    summary = dict(schema='pokezero.set_posterior_device_parity.v1', files=[f.name for f in files],
        table_binding=np.asarray(tables.binding).tolist(), unknown_table_names=unknown, constants_equal=constants,
        counts=dict(sorted(check.stats.items())), mismatches=mismatches, seconds=round(time.time() - t0, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True))
    (out / 'samples.json').write_text(json.dumps(check.samples, indent=1, sort_keys=True, default=str))
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if mismatches == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
