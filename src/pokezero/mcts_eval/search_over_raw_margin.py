"""Prospective conditional margin arithmetic, not collection authority.

Zero-contribution seed clusters stay in ratio bootstraps. Empty-denominator
resamples receive the full outcome range rather than being silently dropped.
Unknown deviation membership is optional membership with an unknown contrast.
"""
import math
import random


def require(condition, message):
    if not condition:
        raise ValueError(message)


def raw_margin(prior, legal):
    require(type(prior) is list and type(legal) is list and legal
        and len(legal)==len(set(legal)) and all(type(a) is int and 0<=a<len(prior) for a in legal)
        and all(type(p) in (int,float) and math.isfinite(p) and p>=0 for p in prior)
        and all(p==0 for a,p in enumerate(prior) if a not in legal)
        and math.isclose(math.fsum(prior),1.,rel_tol=0.,abs_tol=2e-5),
        "original normalized legal raw prior required")
    top=sorted((prior[a] for a in legal),reverse=True)
    return top[0]-top[1] if len(top)>1 else None


def in_band(margin):
    require(margin is None or type(margin) in (int,float)
        and math.isfinite(margin) and 0<=margin<=1.,"finite top-two margin required")
    return margin is not None and .1<=margin<=.3


def member(margin, raw, selected):
    if not in_band(margin):return False
    return None if selected is None else selected!=raw


def ratio_bounds(groups, weighting="pooled_roots"):
    require(weighting in {"pooled_roots","equal_seed"} and groups and all(groups),
        "nonempty scanned source clusters required")
    denominator=unknown=lower=upper=0.
    for group in groups:
        weight=1. if weighting=="pooled_roots" else 1./len(group)
        for row in group:
            membership=row["member"];pair=row["contrast"]
            require(membership is None or type(membership) is bool,"explicit conditional membership required")
            require(len(pair)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in pair)
                and -1<=pair[0]<=pair[1]<=1,"bounded contrast required")
            if membership is None:
                require(tuple(pair)==(-1.,1.),"unknown membership must retain full outcome range")
                unknown+=weight
            elif membership:
                denominator+=weight;lower+=weight*pair[0];upper+=weight*pair[1]
    total=denominator+unknown
    return [-1.,1.] if total==0 else [(lower-unknown)/total,(upper+unknown)/total]


def summarize(rows, *, bootstrap_seed, draws=10000):
    require(type(draws) is int and draws>=100 and type(bootstrap_seed) is int,"frozen bootstrap required")
    require(rows and len({(r["seed"],r["request_index"]) for r in rows})==len(rows),
        "complete unique scanned identities required")
    by_seed={}
    for row in rows:by_seed.setdefault(row["seed"],[]).append(row)
    groups=list(by_seed.values());rng=random.Random(bootstrap_seed)
    estimates={}
    for weighting in ("pooled_roots","equal_seed"):
        boots=[];empty=0
        for _ in range(draws):
            sample=rng.choices(groups,k=len(groups))
            empty+=not any(r["member"] is not False for g in sample for r in g)
            boots.append(ratio_bounds(sample,weighting))
        estimates[weighting]=dict(identification_bounds=ratio_bounds(groups,weighting),
            descriptive_seed_bootstrap_95=[sorted(p[0] for p in boots)[int(.025*draws)],
                sorted(p[1] for p in boots)[int(.975*draws)]],
            descriptive_one_sided_95_lower=sorted(p[0] for p in boots)[int(.05*draws)],
            empty_denominator_resamples=empty,empty_denominator_rule="retain [-1,1]; never discard")
    known=[r for r in rows if r["member"] is True]
    unknown=sum(r["member"] is None for r in rows)
    return dict(scanned_roots=len(rows),scanned_source_clusters=len(groups),
        known_band_deviations=len(known),unknown_membership_roots=unknown,
        deviation_source_clusters=len({r["seed"] for r in known}),
        outcome_identified_band_deviations=sum(r["contrast"][0]==r["contrast"][1] for r in known),
        conditional_mean_status="UNCERTAIN_MEMBERSHIP" if unknown else "DEFINED" if known else "UNDEFINED_NO_DEVIATIONS",
        estimates=estimates,primary_weighting="pooled_roots conditional ratio",
        bootstrap_seed=bootstrap_seed,bootstrap_draws=draws,
        calibration_certified=False,holdout_authorized=False,scientific_strength_claim=False)
