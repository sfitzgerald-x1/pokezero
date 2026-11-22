#!/usr/bin/env python3
"""Check finite completion for an exported catalog using exact Boolean/PB logic.

Requires z3-solver. Writes the actual SMT inputs and a compact result manifest;
unknown/timeout is a failed proof, never an authorization for a shortcut.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prove(model_path, output, *, timeout_ms=120000):
    import z3
    model = json.loads(Path(model_path).read_text())
    payload = {k: v for k, v in model.items() if k != 'sha256'}
    digest = hashlib.sha256(json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    if digest != model['sha256'] or model['slots'] != 6:
        raise ValueError('verified six-slot catalog required')
    bases = model['bases']; names = {b['name']: i for i, b in enumerate(bases)}
    caps = set()
    for field, limit in (('types', 2), ('weak', 3), ('double_weak', 1)):
        for kind in model['types']:
            members = tuple(i for i, b in enumerate(bases) if kind in b[field])
            if members:
                caps.add((members, limit))
    caps.add((tuple(i for i, b in enumerate(bases) if b['level100']), 1))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records = []
    for incoming in (False, True):
        solver = z3.SolverFor('QF_FD'); solver.set(timeout=timeout_ms)
        chosen = [z3.Bool(f'chosen_{i}') for i in range(len(bases))]
        constraints = [z3.PbLe([(v, 1) for v in chosen], 5)]
        saturated = []
        for members, limit in sorted(caps):
            terms = [(chosen[i], 1) for i in members]
            constraints.append(z3.PbLe(terms, limit))
            saturated.append((members, z3.PbEq(terms, limit)))
        for i, base in enumerate(bases):
            for name in base['incompatible']:
                constraints.append(z3.Not(z3.And(chosen[i], chosen[names[name]])))
            if incoming and base['name'] == 'Wobbuffet':
                constraints.append(z3.Not(chosen[i]))
                continue
            # Every remaining available base must be blocked. A SAT witness
            # would be a legal partial team with no possible next member.
            constraints.append(z3.Or(chosen[i], *[sat for members, sat in saturated if i in members],
                                     *[chosen[names[name]] for name in base['incompatible']]))
        solver.add(*constraints)
        recorder = z3.Solver(); recorder.add(*constraints)
        formula = recorder.to_smt2()
        (output / f'completion-{int(incoming)}.smt2').write_text(formula)
        result = solver.check()
        record = dict(initial_wobbuffet=incoming, result=str(result), solver=z3.get_version_string(),
                      model_sha256=digest, formula_sha256=hashlib.sha256(formula.encode()).hexdigest())
        if result == z3.sat:
            witness = solver.model()
            record['witness'] = [b['name'] for i, b in enumerate(bases) if z3.is_true(witness.eval(chosen[i]))]
        elif result == z3.unknown:
            record['reason'] = solver.reason_unknown()
        records.append(record)
    (output / 'completion.json').write_text(json.dumps(records, indent=2) + '\n')
    if any(record['result'] != 'unsat' for record in records):
        raise RuntimeError('completion proof failed; inspect result manifest')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model'); parser.add_argument('output')
    parser.add_argument('--timeout-ms', type=int, default=120000)
    args = parser.parse_args()
    print(json.dumps(prove(args.model, args.output, timeout_ms=args.timeout_ms)))
