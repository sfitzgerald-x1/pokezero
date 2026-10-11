"""Fixed-K chance averaging, not retry-until-match conditioning.

For K IID ORIGINAL chance draws, let M match the observation. Select uniformly
among those M and give the child weight M/K. For every test function f:
 E[(M/K) f(selected)] = E[(1/K) sum_j 1[E_j] f(child_j)]
                         = E_original[1[E] f(child)].
Thus the observation likelihood remains in the weight even when it depends on
hidden team/state. Multiplying by pi(action)/q(action) also corrects guided action
proposals. This is finite-population SMC, not an exact posterior or paper rule.
All K trials must finish. Zero matches returns zero potential; partial/censored
pools refuse. Native errors propagate, never become impossible observations.
"""
import copy
import math

from .paper_reference import ReferenceRefusal
from .paper_reference_particles import WeightedAdvance, release_unique


def validate_chance_pool_count(value, *, particles):
    if (type(value) is not int or not 1 <= value <= 8
            or value != 1 and not particles):
        raise ReferenceRefusal('fixed chance pools require integer 1..8 and explicit particles')
    return value


def fixed_chance_pool(*, count, trial, release, rng, check, receipt):
    if type(count) is not int or not 1 <= count <= 8:
        raise ReferenceRefusal('invalid fixed chance pool count')
    survivors=[]; weights=[]
    receipt.update(requested=count,attempted=0,completed=0,matches=0,
                   pool_complete=False,partial_pool_used=False,observation_likelihood_estimate=None)
    try:
        for ordinal in range(count):
            check()
            receipt['attempted']+=1
            child=trial(ordinal)
            if child is not None:
                weight = child.weight if isinstance(child, WeightedAdvance) else 1.
                child = child.particle if isinstance(child, WeightedAdvance) else child
                survivors.append(child)
                if type(weight) not in (int,float) or not math.isfinite(weight) or weight <= 0:
                    raise ReferenceRefusal('invalid chance proposal importance weight')
                weights.append(weight)
                receipt['matches']+=1
            receipt['completed']+=1
            check()
        total = math.fsum(weights)
        receipt.update(pool_complete=True,observation_likelihood_estimate=total/count,
            survivor_importance_weights=list(weights))
        if not survivors:return None
        index=0 if count==1 else (rng.randrange(len(survivors)) if len(set(weights)) == 1
            else rng.choices(range(len(survivors)), weights=weights,k=1)[0])
        selected=survivors[index]
        receipt['selected_matching_ordinal']=index
        discard=[p for p in survivors if p is not selected]
        survivors=[selected]
        release_unique(discard,release)
        survivors=[]  # ownership of the selected child transfers to the caller
        return WeightedAdvance(selected,receipt['observation_likelihood_estimate'])
    except BaseException as error:
        error.sampling_diagnostic=dict(schema='pokezero.fixed-chance-pool-failure.v1',
            chance_pool=copy.deepcopy(receipt),cause_sampling_diagnostic=copy.deepcopy(
                getattr(error,'sampling_diagnostic',None)))
        raise
    finally:
        release_unique(survivors,release)
