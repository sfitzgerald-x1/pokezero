"""Descriptive full-denominator and predeclared margin readout for 16384 work.

This is the October 18 definitive budget experiment, not another mechanism
probe or holdout screen. Every reservation stays in the overall denominator.
Margin ratios retain zero-contribution source clusters and unknown membership.
"""
from .search_over_raw_margin import require, summarize

BANDS=("below","in_band","above","single_legal_action")


def _ratio(rows,seed,draws):
    result=summarize(rows,bootstrap_seed=seed,draws=draws)
    return dict(known_members=result["known_band_deviations"],
        unknown_membership=result["unknown_membership_roots"],
        contributing_source_clusters=result["deviation_source_clusters"],
        identified_members=result["outcome_identified_band_deviations"],
        ratio_status=result["conditional_mean_status"],estimates=result["estimates"])


def summarize_budget(rows,*,bootstrap_seed,draws=10000):
    if not rows:
        return dict(scanned_roots=0,scanned_source_clusters=0,identified_roots=0,
            estimates={},conditional_deviations={},margin_strata={},candidate_nominated=False)
    for row in rows:
        require(type(row["source_validated"]) is bool and row["band"] in BANDS,"validated source and declared margin band required")
        require(row["source_validated"] or row["member"] is None,"failed source cannot establish deviation membership")
        require(row["member"] is not False or tuple(row["contrast"])==(0.,0.),"known raw alias must have zero contrast")
    # Overall membership is fixed for every actual reservation, even if its
    # source failed validation. Such a root retains its full contrast range.
    overall=summarize([{**r,"member":True} for r in rows],bootstrap_seed=bootstrap_seed,draws=draws)
    strata={}
    for band in BANDS:
        scanned=[];deviations=[]
        for row in rows:
            trusted=row["source_validated"]
            in_stratum=row["band"]==band
            scanned.append({**row,"member":in_stratum if trusted else None})
            deviations.append({**row,"member":row["member"] if in_stratum else False}
                if trusted else {**row,"member":None})
        strata[band]=dict(scanned_contrast=_ratio(scanned,bootstrap_seed,draws),
            conditional_deviations=_ratio(deviations,bootstrap_seed,draws))
    return dict(scanned_roots=len(rows),scanned_source_clusters=overall["scanned_source_clusters"],
        identified_roots=sum(r["contrast"][0]==r["contrast"][1] for r in rows),
        estimates=overall["estimates"],primary_weighting="pooled_roots full scanned denominator",
        secondary_weighting="equal_seed full scanned mean",
        conditional_deviations=_ratio(rows,bootstrap_seed,draws),margin_strata=strata,
        margin_split="below 0.1; inclusive [0.1,0.3]; above 0.3; single legal action separate",
        bootstrap_seed=bootstrap_seed,bootstrap_draws=draws,candidate_nominated=False,
        calibration_certified=False,holdout_authorized=False,scientific_strength_claim=False)
