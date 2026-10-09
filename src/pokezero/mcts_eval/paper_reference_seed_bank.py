"""Cheap, fully supported importance guidance over ORIGINAL integer seeds.

Draw a COMPLETE fixed bank of L original seeds. For any proposal q(i)>0,
E_q[(1/L)/q(I) * f(seed_I) | bank] = sum_i f(seed_i)/L.
This identity holds for wrong, adaptive, or pilot-derived guidance and duplicate
seeds. Averaging banks preserves the actual SHA-mapped integer-seed law, without
assuming it is uniform over the simulator's internal states. All native outcomes
still come from their original seeds and pass the original exact constraints.
Finite bank/SMC/self-normalized approximations remain explicitly disclosed.
"""
import hashlib
import math
import re

from .paper_reference import ReferenceRefusal

LCG_A = 0x5D588B656C078965
LCG_C = 0x00269EC3
MASK64 = (1 << 64) - 1


def native_seed_outputs(seed, count):
    """Pinned Showdown Gen5 next() outputs after the existing SHA seed mapping."""
    if type(seed) is not int or type(count) is not int or not 0 <= count <= 256:
        raise ReferenceRefusal('invalid bounded native seed projection')
    state = int.from_bytes(hashlib.sha256(str(seed).encode('utf-8')).digest()[:8], 'big')
    outputs = []
    for _ in range(count):
        state = (LCG_A * state + LCG_C) & MASK64
        outputs.append(state >> 32)
    return outputs


def guided_bank_choice(seeds, compatible, rng, receipt, guided_fraction=.9):
    if (not seeds or len(compatible) != len(seeds) or any(type(c) is not bool for c in compatible)
            or not 0 <= guided_fraction < 1):
        raise ReferenceRefusal('invalid complete seed bank/proposal')
    length, mass = len(seeds), sum(compatible)
    q = [1/length if not mass else (1-guided_fraction)/length +
         (guided_fraction/mass if hit else 0.) for hit in compatible]
    if any(not math.isfinite(p) or p <= 0 for p in q):
        raise ReferenceRefusal('seed bank lost original support')
    index = rng.choices(range(length), weights=q, k=1)[0]
    weight = (1/length)/q[index]
    receipt.update(bank_complete=True, bank_size=length, compatible_seeds=mass,
        original_seeds=list(seeds), compatible=list(compatible), proposal_probabilities=q,
        selected_bank_index=index, selected_seed=seeds[index], importance_weight=weight,
        original_bank_index_probability=1/length, guided_fraction=guided_fraction,
        original_integer_seed_law_preserved=True, exact_posterior=False)
    return seeds[index], weight


def next_public_hits(history, expected):
    if expected[:len(history)] != history:
        raise ReferenceRefusal('seed guidance history is not an exact public prefix')
    hits, current = [], None
    for line in expected[len(history):]:
        p = line.split('|')
        if p[1] == 'turn': break
        if p[1] == 'move' and len(p) >= 5:
            current = dict(source=p[2][:2], target=p[4][:2], move=re.sub('[^a-z0-9]', '', p[3].lower()), critical=False)
            hits.append(current)
        elif current is not None and p[1] == '-crit' and len(p) >= 3 and p[2][:2] == current['target']:
            current['critical'] = True
        elif current is not None and p[1] == '-damage' and len(p) >= 4 and p[2][:2] == current['target']:
            match = re.match(r'^(\d+)/(\d+)(?:\s|$)', p[3])
            if match: current.update(hp=int(match[1]), maxhp=int(match[2]))
    return hits


def chance_hints(trace, history, expected, *, guide_accuracy=False):
    """Heuristic only. Never extrapolate native critical rounding/boost rules."""
    if type(guide_accuracy) is not bool:
        raise ReferenceRefusal('accuracy guidance requires a boolean opt-in')
    hints = encore_hints(trace, history, expected)
    if guide_accuracy:
        hints.extend(accuracy_hints(trace, history, expected))
    for hit in next_public_hits(history, expected):
        matching = [row for row in trace if isinstance(row.get('damage'), dict) and
            all(row['damage'].get(k) == hit[k] for k in ('source', 'target', 'move'))]
        coin = next((row for row in matching if isinstance(row.get('chance'), dict)
                     and row['chance'].get('numerator') == 1
                     and row['chance'].get('denominator') in (2, 3, 4, 8, 16)), None)
        if coin is not None:
            hints.append(dict(kind='coin', ordinal=coin['ordinal'], numerator=1,
                denominator=coin['chance']['denominator'], desired=hit['critical']))
        roll = next((row for row in matching if row.get('chance') is None and row.get('range') in ([16], [16,None])
                     and 'base_damage' in row['damage']), None)
        if roll is not None and hit.get('maxhp', 100) != 100 and hit.get('maxhp') == roll['damage']['target_max_hp']:
            d = roll['damage']
            if d['critical'] != hit['critical']:
                # Doubling after STAB/type rounding is not native critical
                # damage, even without boost changes. Keep only coin guidance
                # unless an original-seed auxiliary pilot observes this branch.
                continue
            base = d['base_damage']
            target = d['target_hp'] - hit['hp']
            # This is a proposal hint, NOT a replacement damage computation.
            # Native crit boost/ability/queue effects can make it wrong.
            support = [r for r in range(16) if max(1, math.floor(math.floor(base*(100-r))/100)) == target]
            if support: hints.append(dict(kind='damage', ordinal=roll['ordinal'], support=support))
    return hints


def accuracy_hints(trace, history, expected):
    """Full-support proposal hints from observed moves and native Accuracy coins.

    An observed miss may have another cause (e.g. invulnerability), and another
    seed may change event ordering. These are deliberately NOT hard constraints.
    Complete original seed banks and uniform-index/q correction preserve the
    original kernel even with wrong hints. Repeated observed moves or multiple
    matching Accuracy contexts get no hint. Original outcomes and exact history decide
    acceptance; no chance roll or Substitute HP is substituted.
    """
    if expected[:len(history)] != history:
        raise ReferenceRefusal('accuracy guidance history is not an exact public prefix')
    moves = []
    for line in expected[len(history):]:
        parts = line.split('|')
        if len(parts) > 1 and parts[1] == 'turn':
            break
        if len(parts) >= 5 and parts[1] == 'move' and parts[2] and parts[4]:
            moves.append(dict(source_ident=parts[2], target_ident=parts[4],
                move=re.sub('[^a-z0-9]', '', parts[3].lower()), desired='[miss]' not in parts[5:]))
    hints = []
    for move in moves:
        keys = ('source_ident', 'target_ident', 'move')
        if sum(all(other[k] == move[k] for k in keys) for other in moves) != 1:
            continue
        rows = [row for row in trace if isinstance(row.get('accuracy'), dict)
            and all(row['accuracy'].get(k) == move[k] for k in keys)]
        if len(rows) != 1:
            continue
        row = rows[0]
        context, coin = row['accuracy'], row.get('chance')
        numerator = context.get('numerator')
        if (type(numerator) not in (int, float) or not math.isfinite(numerator)
                or not 0 <= numerator <= 100 or context.get('denominator') != 100
                or context.get('source') != move['source_ident'][:2]
                or context.get('target') != move['target_ident'][:2]
                or coin != dict(numerator=numerator, denominator=100)
                or type(row.get('ordinal')) is not int or not 1 <= row['ordinal'] <= 256):
            continue
        hints.append(dict(kind='coin', ordinal=row['ordinal'], numerator=numerator,
            denominator=100, desired=move['desired']))
    return hints


def encore_hints(trace, history, expected):
    """Heuristic roll guidance from ALREADY observed Encore episodes.

    A PP-induced early ending can make the hint wrong. It is never a hard
    constraint: every original seed retains positive proposal probability and
    target/proposal correction. Only a native, context-bound 3..6 roll is guided.
    """
    if expected[:len(history)] != history:
        raise ReferenceRefusal('Encore guidance history is not an exact public prefix')
    suffix = expected[len(history):]
    hints = []
    for row in trace:
        context = row.get('volatile')
        if (not isinstance(context, dict) or context.get('id') != 'encore'
                or row.get('range') != [3, 7] or row.get('chance') is not None
                or type(context.get('after_target_acted')) is not bool):
            continue
        ident, start, paid = context.get('ident'), False, 0
        for line in suffix:
            parts = line.split('|')
            kind = parts[1] if len(parts)>1 else ''
            actor = parts[2] if len(parts)>2 else ''
            if not start:
                if kind == 'turn': break  # pilot cannot start a later turn's lock
                if kind == '-start' and actor == ident and parts[3:4] == ['Encore']:
                    start = True
            elif kind in ('switch','drag','replace','faint') and actor == ident:
                break
            elif kind == 'upkeep': paid += 1
            elif kind == '-end' and actor == ident and parts[3:4] == ['Encore']:
                roll = paid + 1 - int(context['after_target_acted'])
                if 3 <= roll <= 6:
                    hints.append(dict(kind='integer_range', ordinal=row['ordinal'],
                        lower=3, upper=7, support=[roll],
                        basis='observed Encore end; heuristic only, PP ending may differ'))
                break
    return hints


def critical_alignment_hints(trace, history, expected):
    """Request at most one extra native pilot; not an accepted-world retry.

    The auxiliary randomization can have ANY distribution: conditional on its
    trace, the subsequent original-bank/proposal identity still holds. It is
    never transition evidence and does not contribute observation likelihood.
    """
    mismatch = False
    for hit in next_public_hits(history, expected):
        rows = [row for row in trace if isinstance(row.get('damage'), dict) and
            all(row['damage'].get(k) == hit[k] for k in ('source','target','move'))]
        if any('critical' in row['damage'] and row['damage']['critical'] != hit['critical']
               for row in rows):
            mismatch = True
    return ([h for h in chance_hints(trace,history,expected) if h['kind']=='coin']
            if mismatch else [])


def matches_hints(seed, hints):
    if not hints: return True
    values = native_seed_outputs(seed, max(h['ordinal'] for h in hints))
    for h in hints:
        value = values[h['ordinal']-1]
        if h['kind'] == 'coin':
            if ((value*h['denominator'] // (1 << 32)) < h['numerator']) != h['desired']: return False
        elif h['kind'] == 'integer_range':
            if h['lower'] + value*(h['upper']-h['lower']) // (1 << 32) not in h['support']: return False
        elif value*16 // (1 << 32) not in h['support']: return False
    return True


def validate_native_trace(trace, seed):
    if not isinstance(trace, list) or len(trace) > 256:
        raise ReferenceRefusal('missing or unbounded native chance trace')
    outputs = native_seed_outputs(seed, len(trace))
    if any(row.get('ordinal') != i+1 or row.get('value') != outputs[i] for i,row in enumerate(trace)):
        raise ReferenceRefusal('pinned native PRNG trace/projection drift')


def matches_hint_bank(seeds, hints, check=lambda: None):
    """Bit-identical SHA/LCG projection, vectorizing only integer arithmetic."""
    if not hints:
        return [True]*len(seeds)
    count=max(h['ordinal'] for h in hints)
    if type(count) is not int or not 0 <= count <= 256:
        raise ReferenceRefusal('invalid bounded native seed projection')
    import numpy as np
    states=[]
    for index,seed in enumerate(seeds):
        if index%16==0:check()
        if type(seed) is not int:
            raise ReferenceRefusal('invalid bounded native seed projection')
        states.append(int.from_bytes(hashlib.sha256(str(seed).encode('utf-8')).digest()[:8],'big'))
    states=np.asarray(states,dtype=np.uint64)
    values={}
    needed={h['ordinal'] for h in hints}
    for ordinal in range(1,count+1):
        if ordinal%16==1:check()
        # Array uint64 arithmetic wraps modulo 2**64, exactly like MASK64.
        states=states*np.uint64(LCG_A)+np.uint64(LCG_C)
        if ordinal in needed:values[ordinal]=(states >> np.uint64(32)).tolist()
    compatible=[True]*len(seeds)
    for hint in hints:
        for index,value in enumerate(values[hint['ordinal']]):
            if hint['kind']=='coin':
                hit=((value*hint['denominator']//(1 << 32)) < hint['numerator'])==hint['desired']
            elif hint['kind']=='integer_range':
                hit=hint['lower']+value*(hint['upper']-hint['lower'])//(1 << 32) in hint['support']
            else:
                hit=value*16//(1 << 32) in hint['support']
            compatible[index]=compatible[index] and hit
    check()
    return compatible


def draw_guided_seed(rng, hints, check, receipt, bank_size=256, *, seed_bits=64):
    if type(bank_size) is not int or not 1 <= bank_size <= 1024:
        raise ReferenceRefusal('invalid fixed seed bank size')
    if type(seed_bits) is not int or seed_bits not in (32,64):
        raise ReferenceRefusal('invalid original integer seed domain')
    seeds, compatible = [], []
    receipt.update(bank_complete=False, bank_size=bank_size, hints=hints,
        original_seeds=seeds, compatible=compatible,
        likelihood_correction='uniform original bank index / full-support proposal index')
    for index in range(bank_size):
        if index % 16 == 0: check()
        seed = rng.getrandbits(seed_bits)
        seeds.append(seed)
    compatible.extend(matches_hint_bank(seeds,hints,check))
    check()
    return guided_bank_choice(seeds, compatible, rng, receipt)
