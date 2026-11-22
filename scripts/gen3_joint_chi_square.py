#!/usr/bin/env python3
"""Two-sample conditional-team diagnostics, with a familywise alpha of .001."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def condition_digest(facts):
    if (set(facts) - {'evidence', 'fixed_slots', 'initial_wobbuffet', 'terminal_weights'}
            or type(facts['initial_wobbuffet']) is not bool):
        raise ValueError('invalid conditioning facts')
    terminal = facts.get('terminal_weights', [1, 1])
    if len(terminal) != 2 or any(type(w) is not int or w != 1 for w in terminal):
        raise ValueError('reference requires unit terminal likelihoods')
    evidence = sorted(facts['evidence'].items())
    if any(not row or any(type(w) is not int or w not in (0, 1) for w in row) for _, row in evidence):
        raise ValueError('reference requires Boolean evidence likelihoods')
    fixed = []
    for slot, species in facts['fixed_slots'].items():
        if str(slot) not in tuple(map(str, range(6))) or not isinstance(species, str):
            raise ValueError('invalid generation slot')
        fixed.append((int(slot), species))
    canonical = ['gen3-joint-condition-v1', evidence, sorted(fixed), facts['initial_wobbuffet'], terminal]
    return hashlib.sha256(json.dumps(canonical, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def check(model_path, posterior_path, reference_paths):
    import numpy as np
    from scipy.stats import binomtest, chi2_contingency
    model = json.loads(Path(model_path).read_text())
    model_hash = model.pop('sha256')
    if model_hash != hashlib.sha256(json.dumps(model, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest():
        raise ValueError('model digest mismatch')
    posterior = json.loads(Path(posterior_path).read_text())
    if (posterior['schema'] != 'gen3-joint-posterior-sample-v2' or posterior['model_sha256'] != model_hash
            or posterior['condition_sha256'] != condition_digest(posterior['facts'])
            or not posterior['normalizer']['exact'] or posterior['samples'] < 10000):
        raise ValueError('invalid posterior sample')
    counts, seeds, teams, accepted = defaultdict(Counter), set(), 0, 0
    for path in reference_paths:
        ref = json.loads(Path(path).read_text())
        if (ref['schema'] != 'gen3-joint-reference-v2' or ref['showdown_commit'] != model['showdown_commit']
                or ref['condition_sha256'] != condition_digest(ref['facts'])
                or ref['condition_sha256'] != posterior['condition_sha256']
                or ref['executed_sha256'] != model['executed_sha256'] or ref['model_sha256'] != model_hash
                or ref['source_hash'] != posterior['source_hash'] or ref['seed'] in seeds
                or ref['sets'] != 6 * ref['teams']):
            raise ValueError('reference provenance/count mismatch')
        seeds.add(ref['seed']); teams += ref['teams']; accepted += ref['accepted']
        for name, histogram in ref['counts'].items():
            if sum(histogram.values()) != ref['accepted']:
                raise ValueError('incomplete reference histogram')
            counts[name].update(histogram)
    if teams < 1000000 or accepted < 1000 or set(counts) != set(posterior['counts']):
        raise ValueError('insufficient conditional reference or mismatched features')
    samples = posterior['samples']; results = []
    threshold = .001 / (len(counts) + 1)
    for name, reference in sorted(counts.items()):
        candidate = posterior['counts'][name]
        if sum(candidate.values()) != samples:
            raise ValueError('incomplete posterior histogram')
        bins, pooled = [], [0, 0]
        # Pool only using margins: each retained bin has >=5 expected in
        # both independent samples. No category is silently discarded.
        for key in sorted(set(reference) | set(candidate)):
            a, b = reference.get(key, 0), candidate.get(key, 0)
            if (a+b) * min(accepted, samples) / (accepted+samples) < 5:
                pooled[0] += a; pooled[1] += b
            else:
                bins.append([a, b])
        if sum(pooled):
            while bins and sum(pooled) * min(accepted, samples) / (accepted+samples) < 5:
                index = min(range(len(bins)), key=lambda i: sum(bins[i]))
                a, b = bins.pop(index); pooled[0] += a; pooled[1] += b
            bins.append(pooled)
        if len(bins) < 2:
            raise ValueError('uninformative histogram')
        stat, p, df, expected = chi2_contingency(np.array(bins).T, correction=False)
        if expected.min() < 5:
            raise ValueError('underpopulated pooled cell')
        results.append(dict(feature=name, chi2=float(stat), df=int(df), p=float(p), threshold=threshold,
                            bins=len(bins), minimum_expected=float(expected.min()), passed=bool(p >= threshold)))
    probability = Fraction(int(posterior['normalizer']['lower']), int(posterior['quantum']) ** model['slots'])
    acceptance = binomtest(accepted, teams, float(probability))
    results.append(dict(feature='evidence_probability', p=float(acceptance.pvalue), threshold=threshold,
                        exact_probability=str(probability), passed=bool(acceptance.pvalue >= threshold)))
    return dict(teams=teams, generator_sets=teams*6, accepted_reference_teams=accepted, posterior_teams=samples,
                familywise_alpha=.001, model_sha256=model_hash, condition_sha256=posterior['condition_sha256'], results=results,
                passed=all(row['passed'] for row in results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True); parser.add_argument('--posterior', required=True)
    parser.add_argument('--reference', nargs='+', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = check(args.model, args.posterior, args.reference)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)
