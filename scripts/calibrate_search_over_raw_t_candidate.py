"""Create-only SYNTHETIC fixed-128 t/skew comparison; no method adoption."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from calibrate_search_over_raw_joint import SCENARIOS, generate, rate, write_new
from pokezero.mcts_eval.search_over_raw_statistics import ARMS, require
from pokezero.mcts_eval.search_over_raw_t_candidate import (
    TCandidateDesign, adjudicate_t_candidate, adjudicate_t_primary_cells, student_critical,
)


def vector_candidate(lower, upper, refused, design):
    """Only one look at 128; third refusal ends the actual observed prefix."""
    require(isinstance(design, TCandidateDesign), "frozen t design required")
    require(lower.shape == upper.shape and lower.ndim == 3 and lower.shape[1:] == (256, 2)
        and refused.shape == lower.shape[:2], "generator batch shape changed")
    lo, hi = lower[:, :128], upper[:, :128]
    missing = (lo != hi).any(axis=2) | refused[:, :128]
    hit = np.cumsum(missing, axis=1) > 2
    failed = hit.any(axis=1)
    stop = np.where(failed, hit.argmax(axis=1) + 1, 128)
    mean = lo.mean(axis=1)
    centered = lo - mean[:, None, :]
    m2 = (centered ** 2).mean(axis=1)
    uncertain = (lo != hi).any(axis=1)
    variance_upper = np.where(uncertain,
        np.maximum(0., np.maximum(lo ** 2, hi ** 2).sum(axis=1) - 128 * np.maximum(mean, 0.) ** 2) / 127,
        m2 * 128 / 127)
    statistic = np.zeros_like(mean)
    np.divide(np.sqrt(128) * np.maximum(mean, 0.), np.sqrt(variance_upper), out=statistic,
        where=variance_upper > 0)
    constant = lo.max(axis=1) <= hi.min(axis=1)
    skew = np.zeros_like(mean)
    np.divide((centered ** 3).mean(axis=1), m2 ** 1.5, out=skew, where=m2 > 0)
    skew *= np.sqrt(128 * 127) / 126
    skew_unknown = uncertain | (m2 <= 0)
    skew_abstains = (skew_unknown | (np.abs(skew) > .5)) if design.skew_limit is not None else np.zeros_like(uncertain)
    crossed = (mean > 0) & (variance_upper > 0) & ~constant & ~skew_abstains
    crossed &= (statistic > student_critical(127)) & ~failed[:, None]
    # No terminal statistic/gate is inferred from the unobserved failed suffix.
    return dict(crossed=crossed, stop=stop, failed=failed,
        skew_abstains=skew_abstains & ~failed[:, None],
        zero_variance_abstains=((variance_upper <= 0) | constant) & ~failed[:, None])


def parity(lower, upper, refused, design, vector, scores=None, known=None):
    for r in range(min(3, len(lower))):
        stop = int(vector["stop"][r])
        rows = [dict(seed=design.seeds[i], provenance_ok=True, refused=bool(refused[r, i]),
            contrasts={arm: tuple(float(x) for x in (lower[r, i, a], upper[r, i, a]))
                for a, arm in enumerate(ARMS)}) for i in range(stop)]
        scalar = adjudicate_t_candidate(design, rows)
        expected = "REFUSAL_CAP_EXCEEDED" if vector["failed"][r] else "FIXED_CAP_REACHED"
        require(scalar["status"] == expected and scalar["observed_clusters"] == stop
            and list(scalar["nominal_primary_crossings"].values()) == vector["crossed"][r].tolist(),
            "scalar/vector t mismatch")
        if scores is not None:
            cells = []
            for i in range(stop):
                mapping = {}
                for a, arm in enumerate(("raw", *ARMS)):
                    for s, seat in enumerate(("p1", "p2")):
                        accepted = bool(known[r, i, a, s])
                        mapping[(arm, seat)] = dict(status="COMPLETE" if accepted else "REFUSED",
                            validated=accepted, score=float(scores[r, i, a, s]) if accepted else None)
                cells.append(dict(seed=design.seeds[i], primary_cells=mapping,
                    refused=bool(refused[r, i]), provenance_ok=True))
            require(adjudicate_t_primary_cells(design, cells) == scalar, "six-cell t parity mismatch")


def run(output, *, replicates=20000, seed=2026101301):
    require(type(replicates) is int and replicates >= 1000, "at least1000 replicates required")
    require(type(seed) is int and 0 <= seed < 2**64, "uint64 simulation seed required")
    root = Path(__file__).resolve().parent.parent
    inputs = [Path(__file__).resolve(), root / "scripts/calibrate_search_over_raw_joint.py",
        root / "src/pokezero/mcts_eval/search_over_raw_statistics.py",
        root / "src/pokezero/mcts_eval/search_over_raw_t_candidate.py",
        root / "tests/test_search_over_raw_statistics.py", root / "tests/test_search_over_raw_t_candidate.py"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    designs = (TCandidateDesign(tuple(range(128)), None), TCandidateDesign(tuple(range(128)), .5))
    contract = dict(kind="pokezero.synthetic-t-candidate.v1", status="FROZEN_NOT_ADOPTED",
        seed=seed, replicates=replicates, chunk_replicates=1000, scenarios=SCENARIOS,
        designs=[asdict(d) for d in designs], terminal_look=128, family_alpha=.05,
        per_arm_alpha=.025, multiplicity="fixed Bonferroni, NOT Holm", refusal_cap=2,
        critical_value=student_critical(127), input_sha256=hashes,
        scientific_calls=0, historical_outcomes_read=False, collection_authorized=False,
        method_adopted=False, phase_b_authorized=False, calibration_certified=False,
        predeclared_gate="adjusted Fisher-Pearson abs skewness <= .5 independently per arm; no threshold fitting",
        unknown_rule="full interval roster; positive lower mean/upper second moment variance envelope; skew gate abstains on uncertain arm; any constant completion abstains",
        statistical_extension="conservative moment envelope over positive-variance completions with constant-completion abstention, not exact completion optimization or finite-sample Student-t validity",
        assumptions=["independent IID seed clusters, unrestricted within-cluster shared-control dependence",
            "fixed 128-seed terminal look, no interim rejection, no t claim at third refused cluster",
            "a sample-skewness gate does not establish normality or finite-sample validity",
            "generic contrast-erasure/discrete stress is not necessarily realizable six-cell game data",
            "invalid cross-cluster dependence is a negative control, excluded from calibration conclusions"],
        generator_scope="unchanged 19 families from joint driver; both shared raw missing seats preserve known candidate scores",
        comparison_scope="two t rules on common generated cohorts; no selection or adoption based on results",
        runtime_scope="actual synthetic wall/CPU; expected clusters and twelve-game budget only; game hours unmeasured",
        sources=["https://dlmf.nist.gov/8.17", "https://itl.nist.gov/div898/handbook/eda/section3/eda3672.htm"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "contract.json", contract)
    contract_hash = hashlib.sha256((output / "contract.json").read_bytes()).hexdigest()
    wall, cpu = time.perf_counter(), time.process_time()
    results = []
    children = np.random.SeedSequence(seed).spawn(len(SCENARIOS))
    for scenario, child in zip(SCENARIOS, children):
        rng = np.random.default_rng(child)
        totals = [dict(false=0, detected=np.zeros(2, dtype=int), both=0, refusals=0,
            skew=np.zeros(2, dtype=int), zero=np.zeros(2, dtype=int), clusters=0, clusters_squared=0)
            for _ in designs]
        for offset in range(0, replicates, 1000):
            cell_faithful = scenario[1] == "games" and scenario[3] in {"none", "random"}
            data = generate(rng, scenario, min(1000, replicates - offset), original_cells=cell_faithful)
            vectors = []
            for design, counts in zip(designs, totals):
                vector = vector_candidate(*data[:3], design)
                vectors.append(vector)
                if offset == 0:
                    parity(*data[:3], design, vector, *(data[3:] if cell_faithful else ()))
                crossed = vector["crossed"]
                counts["false"] += int((crossed & (np.asarray(scenario[2]) <= 0)).any(axis=1).sum())
                counts["detected"] += crossed.sum(axis=0)
                counts["both"] += int(crossed.all(axis=1).sum())
                counts["refusals"] += int(vector["failed"].sum())
                counts["skew"] += vector["skew_abstains"].sum(axis=0)
                counts["zero"] += vector["zero_variance_abstains"].sum(axis=0)
                counts["clusters"] += int(vector["stop"].sum())
                counts["clusters_squared"] += int((vector["stop"].astype(float) ** 2).sum())
            require(bool(np.all(~vectors[1]["crossed"] | vectors[0]["crossed"])), "gate manufactured t crossing")
            require(bool(np.array_equal(vectors[0]["stop"], vectors[1]["stop"]))
                and bool(np.array_equal(vectors[0]["failed"], vectors[1]["failed"])), "gate changed stopping law")
        for design, counts in zip(designs, totals):
            mean = counts["clusters"] / replicates
            variance = max(0., (counts["clusters_squared"] - replicates * mean ** 2) / (replicates - 1))
            results.append(dict(scenario=scenario[0], outside_assumptions=scenario[1] == "dependent",
                true_advantages=dict(zip(ARMS, scenario[2])), method="plain_t" if design.skew_limit is None else "skew_gated_t",
                familywise_false_positive=rate(counts["false"], replicates),
                primary_detection={arm: rate(int(counts["detected"][i]), replicates) for i, arm in enumerate(ARMS)},
                both_detected=rate(counts["both"], replicates), refusal_cap_stop=rate(counts["refusals"], replicates),
                skew_abstention={arm: rate(int(counts["skew"][i]), replicates) for i, arm in enumerate(ARMS)},
                zero_variance_abstention={arm: rate(int(counts["zero"][i]), replicates) for i, arm in enumerate(ARMS)},
                expected_clusters=mean, clusters_standard_error=float(np.sqrt(variance / replicates)),
                expected_full_cluster_game_budget=12 * mean, projected_sequential_hours=None))
        print(scenario[0] + " completed", flush=True)
    require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items()), "source drift")
    require(hashlib.sha256((output / "contract.json").read_bytes()).hexdigest() == contract_hash, "contract drift")
    result = dict(kind="pokezero.synthetic-t-candidate-result.v1", status="COMPLETE_NOT_ADOPTED",
        input_sha256=hashes, contract_sha256=contract_hash, results=results, numpy_version=np.__version__,
        computation_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu,
        scalar_vector_parity_checked=True, original_cell_parity_checked_for_realizable_laws=True,
        source_hashes_unchanged=True, calibration_certified=False, finite_sample_validity_established=False,
        method_adopted=False, scientific_calls=0, historical_outcomes_read=False,
        collection_authorized=False, phase_b_authorized=False, no_representative_runtime_claim=True)
    write_new(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026101301)
    args = parser.parse_args()
    run(args.output, replicates=args.replicates, seed=args.seed)
