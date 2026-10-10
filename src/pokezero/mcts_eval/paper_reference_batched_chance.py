"""Transport-efficient fixed observation pools under the original seed law.

The pool size is fixed from PUBLIC evidence BEFORE any candidate is drawn: 32
for a round with an observed critical hit, otherwise the ordinary declared K.
All draws count, including after a match. Uniform choice amongst prefix matches
has weight M/K; the selected draw is replayed deterministically for authoritative
current-root validation. If that validation fails its potential is zero, not a
retry amongst other candidates. Native/deadline/partial failures always refuse.
This remains a finite likelihood estimate, not analytical/exact conditioning.
"""
from .paper_reference import ReferenceRefusal
from .paper_reference_particles import WeightedAdvance


def public_chance_pool_size(history, expected, ordinary_count):
    if expected[:len(history)] != history:
        raise ReferenceRefusal('chance allocation history is not an exact observed prefix')
    for line in expected[len(history):]:
        if line.startswith('|-crit|'):
            return 32
        if line.startswith('|turn|'):
            break
    return ordinary_count


def batched_chance_pool(*, factory, snapshot, actions, expected, count, trial, rng, receipt):
    factory.check_sampling_deadline()
    seeds = [rng.getrandbits(64) for _ in range(count)]
    receipt.update(requested=count, attempted=0, completed=0, matches=0,
        pool_complete=False, partial_pool_used=False, observation_likelihood_estimate=None,
        original_chance_seeds=seeds, transport='fixed full native batch; selected original seed replay')
    result = factory.env.conditioning_batch_from_search_snapshot(snapshot, actions,
        chance_seeds=seeds, expected_history=expected,
        deadline_at=factory.sampling_deadline_at, fixed_pool=True)
    receipt.update(attempted=result['consumed'], completed=result['consumed'])
    receipt['first_public_mismatches'] = result.get('first_public_mismatches', [])
    factory.check_sampling_deadline()
    if result['deadline_reached'] or result['consumed'] != count:
        raise ReferenceRefusal('fixed native chance pool incomplete; no partial likelihood')
    matches = result['matching_indices']
    if (len(set(matches)) != len(matches) or any(type(i) is not int or not 0 <= i < count for i in matches)):
        raise ReferenceRefusal('fixed native chance pool has invalid candidate ordinals')
    receipt.update(pool_complete=True, matches=len(matches), observation_likelihood_estimate=len(matches)/count)
    if not matches:
        return None
    selected = matches[rng.randrange(len(matches))]
    receipt['selected_matching_ordinal'] = selected
    child = trial(selected, seeds[selected])
    if child is None:
        receipt['selected_final_potential'] = 0
        return None
    receipt['selected_final_potential'] = 1
    return WeightedAdvance(child, len(matches)/count)
