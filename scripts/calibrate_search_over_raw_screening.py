#!/usr/bin/env python3
"""Prospective synthetic-only calibration of the reconciled staged screening.

No engine, checkpoints, scientific seed allocation, collector or holdout access.
One synthetic seed score represents perfectly dependent roots/continuations
within that cluster; this is not calibration of tree fidelity or runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "src/pokezero/mcts_eval/search_over_raw_screening.py"
spec = importlib.util.spec_from_file_location("screening_calibration_kernel", KERNEL_PATH)
kernel = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kernel
spec.loader.exec_module(kernel)

NAMESPACE = "30ecd1e4-ae62-4769-82ab-a356b3348e62"
SIZES = (32, 64, 128, 200)
REPLICATES = 5000
RNG_SEED = 2026101101
BATCH = 100
OVERLAYS = ("a1_unqualified", "a2_incumbent_model_unqualified",
            "a3_reference_10_unqualified", "incumbent_unavailable",
            "fixed_diagnostic_incomplete", "action_alias_25pct")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def bindings(families_path):
    paths = (Path(__file__), KERNEL_PATH,
             ROOT / "tests/test_search_over_raw_screening_calibration.py",
             Path(families_path).resolve())
    return {str(p): digest(p) for p in paths}


def prospective_contract(families_path):
    families = json.loads(Path(families_path).read_text())["scenarios"]
    if len(families) != 26 or len({f["name"] for f in families}) != 26:
        raise ValueError("the frozen 26-family input is required")
    return dict(
        schema="pokezero.staged-bootstrap-synthetic-calibration.v1",
        input_sha256=bindings(families_path), families_path=str(Path(families_path).resolve()),
        families=families, sizes=list(SIZES), replicates=REPLICATES,
        rng_seed=RNG_SEED, analysis_namespace=NAMESPACE, batch=BATCH,
        bootstrap_draws=10000, lower_order_index=500,
        nominal_alpha_per_arm=.05, bonferroni=False,
        original_family_trials=26 * len(SIZES) * REPLICATES,
        procedural_overlay_trials=len(OVERLAYS) * len(SIZES) * REPLICATES,
        overlays=list(OVERLAYS), overlay_family="shared_control_best_delta_0.10",
        selection="Actual choose_evaluators then choose_ceilings per trial; tie model then smaller ceiling. A1 public/model equals A2 model/1. Independent holdout after both choices frozen.",
        effect_map="hp_fraction/10 has family arm mean; all other public leaf/ceiling combinations have mean minus inferior_config_gap. Null families with no gap tie all combinations.",
        shared_raw="One actual raw score cell per seed/panel reused by both arms and every configuration. Candidate cells generated before contrasts. Separate independent panels for exploration and validation.",
        coupling="Low variance: raw=.5, candidate=.25/.75 with P(.75)=.5+2*effect. Extreme: raw=0 with P=(1+base)/2; best is +/-1; inferior candidates condition on that SAME raw to attain effect, with possible zero contrasts. Discrete nulls: raw=max(0,-D), candidate=raw+D, common D across configurations/arms. Shared control: raw and candidates have .1 draw probability, seed difficulty +/- .15, raw win=.45+difficulty, candidate win=raw win+effect.",
        coupling_extension="Extreme inferior distributions now include zero to permit a real shared raw cell; best marginal retained. Discrete tied nulls share all candidate scores; selection noise stressed in other families. Oracle/fixed-work diagnostic scores are .9 placeholders, not engine/evaluator simulations.",
        missingness="Random refusal independently on raw/candidate action cells, rate as family. Positive-dependent refusal: raw masked when raw==0 with family rate; candidate masked when true candidate-raw>0 with family rate. Raw mask common to all consumers. Entire cell (eight perfectly dependent continuations) unknown. Full denominator; no replacement.",
        stop="Per-seed hazard as family; first stop poisons common suffix. Any exploration stop prevents all stage selection; any attempted validation stop prevents either screening. Stopped trials retained in denominator; prefix/suffix intervals accounted for in full-denominator interval summaries. Validation placeholders with neither arm available are not attempts or observed stops.",
        aliases="Only action_alias_25pct overlay: independent per candidate/seed alias probability .25; candidate becomes the identical raw cell and contrast exactly zero even if raw refused. True selected effect attenuated by .75 and reported accordingly.",
        eligibility="All deployment cells initially accepted=1, QUALIFIED. Overlays mark only named configs accepted=0/unqualified; incumbent_unavailable marks both A2 incumbent leaves unqualified. A1 overlay cannot block A2. Fixed diagnostics use target64, no decision ceiling; incomplete overlay accepted63. Diagnostics cannot select.",
        seed_layer="One synthetic root score per seed (quarter increments realizable with 8 continuation scores); within-seed roots fully dependent. Not the 200-root source generator or engine physics.",
        bootstrap_acceleration="Exact stdlib choices count bank from frozen analysis namespace, shared across arms AND trials; conditional calibration on that bank. Quarter-integer dot products exact in float32 within bounded range, partition[500]/(4*n); parity with actual kernel required before rates.",
        reporting="Per-arm false/true screening, any false, both/any true; unconditional denominator includes unavailable/stopped trials; unavailable arms have no selected truth. Conditional-on-eligible detection reported separately. Wilson95 intervals pointwise and conditional on the frozen bootstrap bank, not simultaneous guarantees. Selected effects/configs and interval widths reported.",
        limits="No finite-sample 5% theorem, approved numerical acceptance tolerance, collection authorization, scientific seed allocation, holdout access, Phase B approval, or collection-runtime projection. Synthetic simulation cost only. Outside-assumption cross-seed dependence labeled separately.",
        no_method_fitting=True, max_runtime_seconds=28800,
        scientific_calls=0, registrations_changed=False, holdout_opened=False,
        collection_authorized=False, confirmatory_strength=False)


def bootstrap_bank(n):
    design = kernel.ScreeningDesign(NAMESPACE, tuple(range(n)))
    rng = random.Random(design.random_seed)
    counts = np.zeros((n, kernel.BOOTSTRAP_DRAWS), dtype=np.float32)
    for b in range(kernel.BOOTSTRAP_DRAWS):
        counts[:, b] = np.bincount(rng.choices(range(n), k=n), minlength=n)
    return counts


def bootstrap_endpoints(lows, bank):
    lows = np.asarray(lows, dtype=np.float64)
    if lows.shape[-1] != bank.shape[0] or not np.all(np.isfinite(lows)):
        raise ValueError("full finite seed vectors required")
    quarters = lows * 4
    if np.any(quarters != np.rint(quarters)) or np.any(np.abs(quarters) > 4):
        raise ValueError("synthetic quarter increments in [-1,1] required")
    sums = quarters.astype(np.float32) @ bank
    return np.partition(sums, kernel.LOWER_ORDER_INDEX, axis=-1)[..., kernel.LOWER_ORDER_INDEX].astype(np.float64) / (4 * bank.shape[0])


def parity_check(bank):
    n = bank.shape[0]
    # Exact-zero boundary, skew, ties, uncertainty and opposing arms.
    fixtures = (np.zeros(n), np.full(n, .25),
                np.where(np.arange(n) % 5 == 0, -1., .25),
                np.where(np.arange(n) % 2, .25, -.25))
    design = kernel.ScreeningDesign(NAMESPACE, tuple(range(n)))
    for low in fixtures:
        other = -low
        expected = kernel.screen(design, {
            "incumbent": tuple((float(x), float(max(x, .5))) for x in low),
            "reference": tuple((float(x), float(max(x, .5))) for x in other)})
        got = bootstrap_endpoints(np.stack((low, other)), bank)
        for a, arm in enumerate(kernel.ARMS):
            exact = expected["arms"][arm]["lower_percentile_endpoint"]
            if got[a] != exact or bool(got[a] > 0) != expected["arms"][arm]["screen_positive"]:
                raise ValueError("accelerated bootstrap differs from production kernel")
    return dict(size=n, fixtures=len(fixtures), exact_endpoint_and_decision=True)


def configuration_effects(family, batch):
    # Axis [trial,arm,leaf(model/hp),ceiling(1/3/10)].
    bases = np.array([family.get("incumbent_mean", family["mean"]), family["mean"]])
    effects = np.broadcast_to(bases[None, :, None, None], (batch, 2, 2, 3)).copy()
    effects -= family.get("inferior_config_gap", 0.)
    effects[:, :, 1, 2] = bases
    return effects


def extreme_probabilities(base, effects):
    p = (1 + base) / 2
    qpos = np.where(effects <= base, (effects + 1 - p) / p, 1.)
    qneg = np.where(effects <= base, 1., (p - effects) / (1 - p))
    if np.any(qpos < 0) or np.any(qpos > 1) or np.any(qneg < 0) or np.any(qneg > 1):
        raise ValueError("unrealizable shared-raw extreme effect")
    return p, qpos, qneg


def draw_cells(family, rng, effects, n):
    """Actual score cells; effects [trial,arm,config] fixed before drawing."""
    batch = effects.shape[0]
    shape = effects.shape + (n,)
    raw_shape = (batch, 1, 1, n)
    kind = family.get("generator")
    if kind == "shared_control":
        win = .45 + rng.choice([-.15, .15], size=raw_shape)
        raw_u = rng.random(raw_shape)
        raw = np.where(raw_u < win, 1., np.where(raw_u < win + .1, .5, 0.))
        candidate_win = win + effects[..., None]
        if np.any(candidate_win < 0) or np.any(candidate_win + .1 > 1):
            raise ValueError("invalid candidate win probability")
        u = rng.random(shape)
        candidate = np.where(u < candidate_win, 1., np.where(u < candidate_win + .1, .5, 0.))
    elif kind == "low_variance":
        raw = np.full(raw_shape, .5)
        p = .5 + 2 * effects[..., None]
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("invalid low-variance effect")
        candidate = np.where(rng.random(shape) < p, .75, .25)
    elif kind == "extreme":
        p, qpos, qneg = extreme_probabilities(family["mean"], effects[..., None])
        raw = (rng.random(raw_shape) >= p).astype(float)
        u = rng.random(shape)
        candidate = np.where(raw == 0, (u < qpos).astype(float), (u >= qneg).astype(float))
    else:
        if kind == "nonidentical":
            p = np.where(np.arange(n) % 2, 1., .6)
            d = np.where(rng.random((batch, 1, 1, n)) < p, .25, -1.)
        else:
            size = (batch, 1, 1, 1 if family.get("cross_seed_shared") else n)
            d = rng.choice(family["values"], p=family["probs"], size=size)
            d = np.broadcast_to(d, raw_shape)
        raw = np.maximum(0., -d)
        candidate = np.broadcast_to(raw + d, shape).copy()
    return raw, candidate


def contrast_intervals(raw, candidate, raw_missing, candidate_missing, aliases):
    lower = np.where(candidate_missing, 0., candidate) - np.where(raw_missing, 1., raw)
    upper = np.where(candidate_missing, 1., candidate) - np.where(raw_missing, 0., raw)
    # An alias names ONE action cell, including its unknown outcome.
    return np.where(aliases, 0., lower), np.where(aliases, 0., upper)


def draw_panel(family, rng, effects, n, overlay="none"):
    raw, candidate = draw_cells(family, rng, effects, n)
    aliases = rng.random(candidate.shape) < (.25 if overlay == "action_alias_25pct" else 0.)
    candidate = np.where(aliases, raw, candidate)
    raw_missing = rng.random(raw.shape) < family.get("random_missing", 0.)
    raw_missing |= (raw == 0) & (rng.random(raw.shape) < family.get("positive_missing", 0.))
    candidate_missing = rng.random(candidate.shape) < family.get("random_missing", 0.)
    candidate_missing |= (candidate - raw > 0) & (rng.random(candidate.shape) < family.get("positive_missing", 0.))
    stop = rng.random(raw.shape) < family.get("stop_probability", 0.)
    suffix = np.maximum.accumulate(stop, axis=-1)
    raw_missing |= suffix
    candidate_missing |= suffix
    low, high = contrast_intervals(raw, candidate, raw_missing, candidate_missing, aliases)
    return low, high, np.any(stop, axis=(1, 2, 3))


def select_trial(exploration_means, overlay="none", stopped=False):
    """Actual production stage functions, not a flat max over configurations."""
    if stopped:
        return (("incumbent", None, None), ("reference", None, None))
    means = np.asarray(exploration_means).reshape(2, 2, 3)
    a2 = []
    for a, arm in enumerate(kernel.ARMS):
        for l, leaf in enumerate(kernel.DEPLOYABLE_LEAVES):
            unqualified = (overlay == "incumbent_unavailable" and a == 0) or (
                overlay == "a2_incumbent_model_unqualified" and a == 0 and l == 0)
            a2.append(kernel.StageSummary("A2", arm, "public", leaf, 1.,
                float(means[a, l, 0]), 0 if unqualified else 1,
                "UNQUALIFIED_AT_CEILING" if unqualified else "QUALIFIED"))
        # Placebo diagnostics can outperform every deployment score but cannot select.
        a2.append(kernel.StageSummary("A2", arm, "oracle", "model", 1., .9, 1,
            "QUALIFIED", "wall_clock_diagnostic"))
        for leaf in (*kernel.DEPLOYABLE_LEAVES, "raw_rollout"):
            incomplete = overlay == "fixed_diagnostic_incomplete"
            a2.append(kernel.StageSummary("A2", arm, "public", leaf, None, .9,
                63 if incomplete else 64,
                "DIAGNOSTIC_INCOMPLETE" if incomplete else "QUALIFIED", "fixed_work_diagnostic"))
    evaluators = kernel.choose_evaluators(a2)
    a3 = []
    for a, (arm, leaf) in enumerate(evaluators):
        if leaf is None:
            continue
        l = kernel.DEPLOYABLE_LEAVES.index(leaf)
        for c, seconds in enumerate((1., 3., 10.)):
            unqualified = overlay == "a3_reference_10_unqualified" and a == 1 and c == 2
            a3.append(kernel.StageSummary("A3", arm, "public", leaf, seconds,
                float(means[a, l, c]), 0 if unqualified else 1,
                "UNQUALIFIED_AT_CEILING" if unqualified else "QUALIFIED"))
    return kernel.choose_ceilings(evaluators, a3)


def wilson(successes, total):
    if not total:
        return dict(count=0, denominator=0, rate=None, mc_interval95=None)
    p, z = successes / total, 1.959963984540054
    denom = 1 + z*z / total
    middle = (p + z*z / (2*total)) / denom
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denom
    return dict(count=int(successes), denominator=int(total), rate=p,
                mc_interval95=[max(0., middle-radius), min(1., middle+radius)])


def run_condition(family, n, bank, rng, overlay="none", replicates=REPLICATES, deadline=None):
    records = []
    for start in range(0, replicates, BATCH):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("synthetic calibration runtime bound reached; no complete result")
        batch = min(BATCH, replicates-start)
        effects = configuration_effects(family, batch).reshape(batch, 2, 6)
        low, high, exploration_stop = draw_panel(family, rng, effects, n, overlay)
        # A1 public/model is the exact A2 model/1 cell. A1's independent
        # diagnostic failure is recorded, never supplied as an A2 prerequisite.
        choices = [select_trial(low[t].mean(axis=-1), overlay, bool(exploration_stop[t]))
                   for t in range(batch)]
        selected = np.zeros((batch, 2, 1))
        available = np.zeros((batch, 2), dtype=bool)
        selected_config = []
        for t, choice in enumerate(choices):
            names = []
            for a, (_, leaf, seconds) in enumerate(choice):
                if leaf is None:
                    names.append("UNAVAILABLE")
                    continue
                index = 3 * kernel.DEPLOYABLE_LEAVES.index(leaf) + (1., 3., 10.).index(seconds)
                selected[t, a, 0] = effects[t, a, index]
                available[t, a] = True
                names.append(f"{leaf}/{int(seconds)}")
            selected_config.append(names)
        # A new independent panel generated only after both choices frozen.
        val_low, val_high, validation_stop = draw_panel(family, rng, selected, n, overlay)
        validation_attempted = available.any(axis=1)
        validation_stop &= validation_attempted
        val_low, val_high = val_low[:, :, 0], val_high[:, :, 0]
        val_low = np.where(available[:, :, None], val_low, -1.)
        val_high = np.where(available[:, :, None], val_high, 1.)
        endpoint = bootstrap_endpoints(val_low, bank)
        positive = (endpoint > 0) & available & ~validation_stop[:, None]
        truth = selected[:, :, 0] * (.75 if overlay == "action_alias_25pct" else 1.)
        for t in range(batch):
            records.append(dict(positive=positive[t], available=available[t], truth=truth[t],
                choice=selected_config[t], exploration_stop=bool(exploration_stop[t]),
                validation_stop=bool(validation_stop[t]),
                validation_attempted=bool(validation_attempted[t]),
                uncertainty=(val_low[t] != val_high[t]).mean(axis=-1),
                width=(val_high[t] - val_low[t]).mean(axis=-1)))
    positives = np.stack([r["positive"] for r in records])
    available = np.stack([r["available"] for r in records])
    truth = np.stack([r["truth"] for r in records])
    true_selected = available & (truth > 0)
    false_selected = available & (truth <= 0)
    detected = positives & true_selected
    falsely_screened = positives & false_selected
    arms = {}
    for a, arm in enumerate(kernel.ARMS):
        distribution = {}
        for row in records:
            choice = row["choice"][a]
            distribution[choice] = distribution.get(choice, 0) + 1
        eligible = true_selected[:, a]
        arms[arm] = dict(false_screening=wilson(int(falsely_screened[:, a].sum()), replicates),
            true_screening=wilson(int(detected[:, a].sum()), replicates),
            detection_given_positive_selected=wilson(int(detected[:, a].sum()), int(eligible.sum())),
            unavailable=wilson(int((~available[:, a]).sum()), replicates),
            selected_configs=distribution,
            selected_effect_counts=[dict(effect=float(e), count=int(np.sum(available[:, a] & (truth[:, a] == e))))
                for e in sorted(set(truth[available[:, a], a]))],
            mean_uncertain_seed_fraction=float(np.mean([r["uncertainty"][a] for r in records])),
            mean_identification_width=float(np.mean([r["width"][a] for r in records])))
    return dict(family=family["name"], overlay=overlay, seeds=n, trials=replicates,
        outside_assumptions=family.get("outside_assumptions", False), arms=arms,
        any_false_screening=wilson(int(falsely_screened.any(axis=1).sum()), replicates),
        any_true_screening=wilson(int(detected.any(axis=1).sum()), replicates),
        both_true_screening=wilson(int(detected.all(axis=1).sum()), replicates),
        exploration_stop=wilson(sum(r["exploration_stop"] for r in records), replicates),
        validation_stop=wilson(sum(r["validation_stop"] for r in records), replicates),
        validation_attempted=wilson(sum(r["validation_attempted"] for r in records), replicates),
        validation_stop_given_attempt=wilson(sum(r["validation_stop"] for r in records),
                                             sum(r["validation_attempted"] for r in records)),
        a1_unqualified_trials=replicates if overlay == "a1_unqualified" else 0,
        fixed_diagnostic_incomplete_trials=replicates if overlay == "fixed_diagnostic_incomplete" else 0)


def validate_contract(contract):
    expected = prospective_contract(contract["families_path"])
    if contract != expected:
        raise ValueError("prospective contract/input/code binding mismatch")


def run(contract_path, review_path, output_path):
    contract = json.loads(Path(contract_path).read_text())
    validate_contract(contract)
    review = json.loads(Path(review_path).read_text())
    if review.get("contract_sha256") != digest(contract_path) or review.get("status") != "CLEARED_SYNTHETIC_ONLY":
        raise ValueError("independent prelaunch review bound to exact contract required")
    if Path(output_path).exists():
        raise FileExistsError("calibration outputs are create-only; no rerun/overwrite")
    started, cpu_started = time.monotonic(), time.process_time()
    deadline = started + contract["max_runtime_seconds"]
    rng = np.random.default_rng(RNG_SEED)
    results, parity = [], []
    for n in SIZES:
        bank = bootstrap_bank(n)
        parity.append(parity_check(bank))
        cases = [(f, "none") for f in contract["families"]]
        base = next(f for f in contract["families"] if f["name"] == contract["overlay_family"])
        cases += [(base, overlay) for overlay in OVERLAYS]
        for family, overlay in cases:
            condition_started = time.monotonic()
            result = run_condition(family, n, bank, rng, overlay, deadline=deadline)
            result["wall_seconds"] = time.monotonic() - condition_started
            results.append(result)
            # Progress only: do not emit rates while the frozen run is executing.
            print(json.dumps(dict(completed_conditions=len(results), total_conditions=128,
                seeds=n, family=family["name"], overlay=overlay,
                elapsed_seconds=time.monotonic()-started)), flush=True)
    output = dict(schema=contract["schema"] + ".results", contract_sha256=digest(contract_path),
        review_sha256=digest(review_path), input_sha256=contract["input_sha256"],
        results=results, parity=parity, total_trials=sum(r["trials"] for r in results),
        wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
        python=sys.version, numpy=np.__version__, bootstrap_bank_conditioned=True,
        calibration_complete=True, calibration_certified=False, confirmatory_strength=False,
        scientific_calls=0, collection_authorized=False, holdout_opened=False,
        approved_numerical_acceptance_tolerance=None)
    save_new(output_path, output)
    return dict(output=str(Path(output_path).resolve()), total_trials=output["total_trials"],
                wall_seconds=output["wall_seconds"], cpu_seconds=output["cpu_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--families", required=True)
    prepare.add_argument("--output", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--contract", required=True)
    execute.add_argument("--review", required=True)
    execute.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        save_new(args.output, prospective_contract(args.families))
        print(json.dumps(dict(contract=str(Path(args.output).resolve()), sha256=digest(args.output), rates_run=False)))
    else:
        print(json.dumps(run(args.contract, args.review, args.output)))


if __name__ == "__main__":
    main()
