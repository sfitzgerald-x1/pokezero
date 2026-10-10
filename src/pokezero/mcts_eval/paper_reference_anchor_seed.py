"""Full-support guidance, never a hard gender assignment or a native override.

For each complete original 32-bit seed bank B and any heuristic q(i|B,D)>0,
E_q[(1/|B|)/q(I) * f(D,B[I]) | B,D] = mean_B f(D,B[i]).
Static details and the whole history are still checked natively. Rejection
conditioning of the entire anchor proposal contributes one common normalizer;
the remaining seed-index ratio must be applied once at the first SMC advance.
Wrong constructor-call hints can hurt speed, but cannot remove seed support.
Finite-bank/SMC self-normalization remains approximate, not exact posterior.
"""
import math
from .paper_reference import ReferenceRefusal
from .paper_reference_seed_bank import draw_guided_seed
from ..randbat import canonical_gen3_randbat_species_id


def gender_seed_hints(teams, source, details, opponent):
    desired={row['species']:row['gender'] for row in details if row['gender'] in ('M','F')}
    ordinal=0;hints=[]
    for side in ('p1','p2'):
        for mon in teams[side]:
            species=canonical_gen3_randbat_species_id(mon.species)
            fixed=source.species_metadata.get(species,{}).get('gender')
            if mon.gender in ('M','F','N') or fixed in ('M','F','N'):
                continue
            ordinal+=1
            if side==opponent and species in desired:
                hints.append(dict(kind='coin',ordinal=ordinal,denominator=2,numerator=1,
                    desired=desired[species]=='M',basis='hypothetical constructor gender call; heuristic only'))
    return hints


def guided_anchor_seed(rng, teams, source, details, opponent, check):
    hints=gender_seed_hints(teams,source,details,opponent)
    if not hints:return rng.getrandbits(32),None
    receipt=dict(schema='pokezero.original-anchor-seed-proposal.v1',
        original_seed_bits=32,original_seed_support_preserved=True,
        native_gender_overwritten=False,observed_details=details,
        weighting_scope='first complete history-particle advance, once per initial anchor')
    seed,weight=draw_guided_seed(rng,hints,check,receipt,seed_bits=32)
    if not math.isfinite(weight) or weight<=0:
        raise ReferenceRefusal('anchor seed proposal produced invalid weight')
    return seed,receipt


def initial_anchor_weight(anchor):
    receipt=anchor.get('materialization_seed_proposal')
    if receipt is None:return 1.
    weight=receipt.get('importance_weight')
    if (receipt.get('schema')!='pokezero.original-anchor-seed-proposal.v1'
            or receipt.get('original_seed_bits')!=32 or receipt.get('bank_complete') is not True
            or receipt.get('original_seed_support_preserved') is not True
            or receipt.get('native_gender_overwritten') is not False
            or type(weight) not in (int,float) or not math.isfinite(weight) or weight<=0):
        raise ReferenceRefusal('initial anchor seed lost original-support/weight provenance')
    return weight
