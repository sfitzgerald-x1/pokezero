"""Tests for `scripts/value_head_ece.py`, which had none.

The script's whole claim is that the ECE it recomputes IS the one in the published artifacts:
it says so in its docstring, it bootstraps that number, and every paired delta a Phase 3
advancement decision reads is a difference of it. That claim rests on one line of arithmetic
being a faithful transcription of `pokezero.value_calibration`, and it was not: the script
divided by a precomputed bin width (`int((p + 1) / (2 / bins))`) where the library multiplies
by the bin count (`int(((p + 1) / 2) * bins)`). `2 / 10` is not representable in binary
floating point, so the two disagree at 3 of the 9 interior bin edges of the default grid and
put those points in the bin BELOW -- silently, on a metric whose entire purpose is
comparability with numbers computed elsewhere.

So the tests here are agreement tests against the library, driven through the library's own
accumulator rather than through a model, plus hand-computed known answers for the aggregation
and the one formatting path that could not render the value its own metric is defined to
return.

AND `main()`, which had none of that. Pinning the arithmetic left every DECISION the script
makes unpinned: swapping the `BETTER than ref` / `WORSE than ref` labels -- the sentence a
Phase 3 advancement decision is read off -- stayed green, as did deleting the unconditional
recomputed-vs-library cross-check (this file's entire stated guarantee), deleting the
identical-returns-column refusal that is what makes the bootstrap PAIRED rather than merely
simultaneous, drawing a separate bootstrap index per head, and deleting the `--expect-ece`
refusal. None of those need a checkpoint or a GPU: what `main()` consumes is a
(prediction, return) column per head and a library number to compare against, so the loader,
the library and the per-example dump become pure data.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import random
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BINS = 10


def _load():
    spec = importlib.util.spec_from_file_location(
        "value_head_ece", REPO / "scripts" / "value_head_ece.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vhe = _load()


def _library_totals(bins=BINS):
    from pokezero.value_calibration import _ValueCalibrationTotals
    return _ValueCalibrationTotals(bin_count=bins)


def _library_bin_index(prediction, bins=BINS):
    return _library_totals(bins)._bin_index(prediction)


def _edges(bins=BINS):
    return [-1.0 + 2.0 * k / bins for k in range(bins + 1)]


def test_bin_index_at_the_three_edges_the_shipped_arithmetic_got_wrong():
    """Hardcoded, not mirrored off the library: these are the answers, and they are the ones
    the script disagreed with.

    Bins are half-open `[lower, upper)` (`_BinTotals.to_bin` reports them that way), so an
    edge value belongs to the bin ABOVE. The shipped `int((p + 1) / (2 / bins))` returned
    2, 5 and 6 here.
    """
    assert vhe.bin_index(-0.40, BINS) == 3
    assert vhe.bin_index(0.20, BINS) == 6
    assert vhe.bin_index(0.40, BINS) == 7
    assert [_library_bin_index(p) for p in (-0.40, 0.20, 0.40)] == [3, 6, 7]


def test_bin_index_agrees_with_the_library_at_every_bin_edge():
    for bins in (2, 4, 10, 16, 20):
        for p in _edges(bins):
            assert vhe.bin_index(p, bins) == _library_bin_index(p, bins), (bins, p)


def test_bin_index_agrees_with_the_library_over_a_dense_sweep():
    """Edges are where the two disagreed, so the sweep is deliberately edge-heavy: a uniform
    random sample over [-1, 1] never lands on one and reports zero mismatches."""
    rng = random.Random(20260816)
    probes = [k / 1000.0 for k in range(-1000, 1001)]
    probes += [k / 20.0 for k in range(-20, 21)]
    probes += [rng.uniform(-1.0, 1.0) for _ in range(4000)]
    for bins in (2, 10, 20):
        for p in probes:
            assert vhe.bin_index(p, bins) == _library_bin_index(p, bins), (bins, p)


def test_bin_index_closes_the_top_bin_and_clips_out_of_range_predictions():
    """The library's two special cases, both of which keep a saturated head in range."""
    assert vhe.bin_index(1.0, BINS) == BINS - 1 == _library_bin_index(1.0)
    assert vhe.bin_index(1.5, BINS) == BINS - 1 == _library_bin_index(1.5)
    assert vhe.bin_index(-1.0, BINS) == 0 == _library_bin_index(-1.0)
    assert vhe.bin_index(-1.5, BINS) == 0 == _library_bin_index(-1.5)


def test_ece_from_reproduces_the_librarys_ece_bias_and_mae():
    """The cross-check the script performs at runtime, performed here without a checkpoint.

    `_ValueCalibrationTotals` is the same accumulator `evaluate_value_calibration` feeds, so
    driving it with a synthetic (prediction, return) column is the library's own number. The
    column includes every bin edge many times over, because that is the only place the two
    implementations ever differed.
    """
    rng = random.Random(4242)
    preds, rets = [], []
    for _ in range(2000):
        preds.append(rng.uniform(-1.0, 1.0))
        rets.append(rng.choice((-1.0, 0.0, 1.0)))
    for _ in range(40):
        for edge in _edges():
            preds.append(edge)
            rets.append(rng.choice((-1.0, 0.0, 1.0)))

    got = vhe.ece_from(preds, rets, BINS)
    totals = _library_totals()
    totals.add(predictions=tuple(preds), returns=tuple(rets))
    report = totals.to_report()

    assert got["n"] == report.examples
    assert got["ece"] == pytest.approx(report.expected_calibration_error, abs=1e-12)
    assert got["bias"] == pytest.approx(report.bias, abs=1e-9)
    assert got["mae"] == pytest.approx(report.mae, abs=1e-9)
    # The disagreement this test exists to catch is ~1e-4 on this column, not 1e-12: the
    # tolerance is tight because the two are meant to be the same arithmetic.
    assert got["ece"] > 0.05


def test_ece_from_is_a_hand_computed_known_answer():
    """Three examples, two bins, weighted by occupancy.

    -0.9 lands in bin 0 twice (mean prediction -0.9, mean return -0.9, so that bin contributes
    nothing however wrong the individual examples are -- which is the property ECE has and MAE
    does not), and 0.5 lands alone in bin 7 with an error of 0.4 at weight 1/3.
    """
    preds = [-0.9, -0.9, 0.5]
    rets = [-1.0, -0.8, 0.1]
    got = vhe.ece_from(preds, rets, BINS)
    assert [vhe.bin_index(p, BINS) for p in preds] == [0, 0, 7]
    assert got["ece"] == pytest.approx(0.4 / 3.0)
    assert got["bias"] == pytest.approx(0.4 / 3.0)
    assert got["mae"] == pytest.approx(0.6 / 3.0)
    assert got["abs_bias"] == pytest.approx(0.4 / 3.0)
    assert got["bias_share_of_ece"] == pytest.approx(1.0)
    assert got["n"] == 3
    # MAE is 0.2 while ECE is 0.133: the two examples that cancel inside bin 0 are exactly the
    # difference, so a test asserting only one of them would not see a binning change.
    assert got["mae"] > got["ece"]


def test_bias_share_of_ece_is_absent_when_ece_is_zero_and_still_renders():
    """`abs(bias)/ece` is undefined at ECE 0 -- a head perfectly calibrated on this set, which
    is the one head nobody would want the tool to crash on. Both readouts formatted it with
    `:.4f`, which raises TypeError on None.
    """
    got = vhe.ece_from([0.25, -0.25], [0.25, -0.25], BINS)
    assert got["ece"] == 0.0
    assert got["bias"] == 0.0
    assert got["bias_share_of_ece"] is None
    with pytest.raises(TypeError):
        f"{got['bias_share_of_ece']:.4f}"          # the formatting that shipped
    assert vhe.fmt_share(got["bias_share_of_ece"]) == f"{'n/a':>11s}"
    assert vhe.fmt_share(None, 0) == "n/a"
    assert vhe.fmt_share(0.5) == f"{0.5:11.4f}"
    assert vhe.fmt_share(0.5, 0) == "0.5000"


def test_the_two_ece_tolerances_are_distinct_and_the_help_says_which_is_unconditional():
    """`--expect-ece`'s help claimed its 1e-6 comparison was what protected comparability with
    the artifacts. It is not: the recomputed-vs-library cross-check runs on every head whether
    or not the flag is passed, and at 1e-9. Two different checks with different subjects were
    described as one, so a reader who skipped the optional flag believed nothing was verifying
    that the bootstrapped metric is the published metric.
    """
    assert vhe.LIBRARY_CROSSCHECK_TOL == 1e-9
    assert vhe.PUBLISHED_ECE_TOL == 1e-6
    assert vhe.LIBRARY_CROSSCHECK_TOL < vhe.PUBLISHED_ECE_TOL
    # The help text is BUILT from the constants, so it cannot drift from what runs.
    assert f"{vhe.LIBRARY_CROSSCHECK_TOL:g}" in vhe.EXPECT_ECE_HELP
    assert f"{vhe.PUBLISHED_ECE_TOL:g}" in vhe.EXPECT_ECE_HELP
    assert "unconditional" in vhe.EXPECT_ECE_HELP
    assert "SEPARATE check" in vhe.EXPECT_ECE_HELP


def test_pct_indexes_the_sorted_draws_and_cannot_run_off_the_end():
    draws = [0.5, 0.1, 0.9, 0.3, 0.7]
    assert vhe.pct(draws, 0.0) == 0.1
    assert vhe.pct(draws, 0.5) == 0.5
    assert vhe.pct(draws, 1.0) == 0.9


# ---------------------------------------------------------------------------------------
# main(), driven end to end against pure data.
#
# The checkpoint loader, the calibration library and the per-example dump are the only heavy
# dependencies, and all three are replaceable by stubs -- the same `sys.modules` technique the
# rescore tests use. What is left is exactly the part that had no coverage: the two refusals
# that guard comparability, the pairing of the bootstrap, and the verdict.
# ---------------------------------------------------------------------------------------

N_EXAMPLES = 300


def _returns_column(seed=20260816, n=N_EXAMPLES):
    """The shared ground-truth column every head is scored against."""
    rng = random.Random(seed)
    return [rng.choice((-1.0, 0.0, 1.0)) for _ in range(n)]


def _preds(rets, *, gain, offset, jitter, seed):
    """A head's predictions: a gain/offset/jitter view of the returns, clipped to [-1, 1].

    `gain` under 1 is a compressed head, `offset` is the constant bias that cancels in a
    sibling comparison, `jitter` is the per-position scatter that does not.
    """
    rng = random.Random(seed)
    return [min(1.0, max(-1.0, gain * r + offset + rng.uniform(-jitter, jitter)))
            for r in rets]


RETS = _returns_column()
# ECE 0.321 / 0.034 / 0.619: `good` is unambiguously better calibrated than `ctl` and `bad`
# unambiguously worse, so the two verdicts below are known answers rather than coin flips.
CTL_PREDS = _preds(RETS, gain=0.5, offset=0.10, jitter=0.25, seed=11)
GOOD_PREDS = _preds(RETS, gain=1.0, offset=0.0, jitter=0.10, seed=12)
BAD_PREDS = _preds(RETS, gain=0.2, offset=0.65, jitter=0.25, seed=13)


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _run_main(monkeypatch, tmp_path, head_preds, *, ref="ctl", rets=None, per_head_rets=None,
              library_ece=None, argv_extra=(), boot=300, boot_seeds="1,2,3"):
    """Run `value_head_ece.main()` over in-memory columns.

    `head_preds` maps head name -> prediction column, in COMMAND-LINE order. `per_head_rets`
    gives a head a different returns column (the thing the pairing refusal exists to catch);
    `library_ece` makes the stubbed library disagree with the recomputation (the thing the
    unconditional cross-check exists to catch). Both default to "everything agrees".
    """
    rets = list(RETS if rets is None else rets)
    per_head_rets = dict(per_head_rets or {})
    data_dir = tmp_path / "calibration-shard"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "calibration.bin").write_bytes(b"frozen calibration rows")
    for name in head_preds:
        (tmp_path / f"{name}.pt").write_bytes(f"checkpoint:{name}".encode())
    json_out = tmp_path / "ece.json"

    def rets_for(name):
        return list(per_head_rets.get(name, rets))

    def load_transformer_checkpoint(path, map_location=None):
        name = Path(path).stem
        assert name in head_preds, name
        return types.SimpleNamespace(name=name), types.SimpleNamespace(name=name)

    def dump(model, result, data, batch_size, device):
        assert [str(p) for p in data] == [str(data_dir)]
        return list(head_preds[model.name]), rets_for(model.name)

    def evaluate_value_calibration(*, model, training_result, paths, batch_size, bins, device):
        if library_ece and model.name in library_ece:
            return types.SimpleNamespace(
                expected_calibration_error=library_ece[model.name])
        return types.SimpleNamespace(expected_calibration_error=vhe.ece_from(
            head_preds[model.name], rets_for(model.name), bins)["ece"])

    monkeypatch.setattr(vhe, "_dump", dump)
    monkeypatch.setitem(sys.modules, "pokezero.neural_policy", _module(
        "pokezero.neural_policy",
        load_transformer_checkpoint=load_transformer_checkpoint))
    monkeypatch.setitem(sys.modules, "pokezero.value_calibration", _module(
        "pokezero.value_calibration",
        evaluate_value_calibration=evaluate_value_calibration))

    argv = ["value_head_ece.py", "--ref", ref, "--data", str(data_dir),
            "--bins", str(BINS), "--boot", str(boot), "--boot-seeds", boot_seeds,
            "--device", "cpu", "--json", str(json_out)]
    for name in head_preds:
        argv += ["--head", f"{name}={tmp_path / (name + '.pt')}"]
    argv += list(argv_extra)
    monkeypatch.setattr(sys, "argv", argv)

    try:
        status, message = vhe.main(), ""
    except SystemExit as exc:
        status, message = (1 if exc.code else 0), ("" if exc.code is None else str(exc.code))
    doc = json.loads(json_out.read_text()) if json_out.exists() else None
    return types.SimpleNamespace(status=status, message=message, json=doc)


def _verdict_lines(captured):
    """The `d ECE` line per non-ref head, keyed by head name."""
    return {ln.split()[0]: ln for ln in captured.splitlines() if "d ECE" in ln}


def test_main_labels_a_lower_ece_head_better_and_a_higher_ece_head_worse(monkeypatch,
                                                                        tmp_path, capsys):
    """The verdict, tied to the DIRECTION of the delta on the same line.

    Swapping the `BETTER than ref` and `WORSE than ref` strings left all 43 tests green. This
    is the sentence a Phase 3 advancement decision is read off, so it is asserted against a
    known answer: `good` recovers the returns almost exactly (ECE 0.034) and `bad` is offset by
    +0.65 (ECE 0.619) against a `ctl` at 0.321. Both paired intervals are far from zero, so
    neither verdict is a coin flip.
    """
    got = _run_main(monkeypatch, tmp_path,
                    {"ctl": CTL_PREDS, "good": GOOD_PREDS, "bad": BAD_PREDS})
    out = capsys.readouterr().out
    assert got.status == 0, got.message

    lines = _verdict_lines(out)
    assert set(lines) == {"good", "bad"}, out
    # The label and the sign are asserted TOGETHER: either one alone is satisfied by the swap.
    assert "d ECE -" in lines["good"] and "BETTER than ref" in lines["good"]
    assert "d ECE +" in lines["bad"] and "WORSE than ref" in lines["bad"]
    assert "WORSE" not in lines["good"] and "BETTER" not in lines["bad"]

    lo, hi = got.json["paired_delta_ece_ci95_vs_ref"]["good"]
    assert hi < 0, "a better-calibrated head's whole delta interval is below zero"
    blo, bhi = got.json["paired_delta_ece_ci95_vs_ref"]["bad"]
    assert blo > 0
    assert got.json["heads"]["good"]["ece"] < got.json["heads"]["ctl"]["ece"] \
        < got.json["heads"]["bad"]["ece"]


def test_main_calls_a_delta_that_straddles_zero_within_noise(monkeypatch, tmp_path, capsys):
    """The third verdict, so the two above cannot be satisfied by always deciding.

    A head whose predictions ARE the reference's has a paired delta of exactly zero at every
    rep, which is the one case where "within noise" is a known answer rather than a threshold.
    """
    got = _run_main(monkeypatch, tmp_path, {"ctl": CTL_PREDS, "twin": list(CTL_PREDS)})
    out = capsys.readouterr().out
    assert got.status == 0, got.message
    assert "within noise" in _verdict_lines(out)["twin"]
    assert "BETTER" not in out and "WORSE" not in out


def test_main_bootstrap_is_paired_over_one_resampled_index_set(monkeypatch, tmp_path):
    """A head identical to the reference must have a delta interval of EXACTLY zero.

    That is only true if the resampling happens ONCE and the same example indices are applied
    to every head. Drawing a separate index per head -- the mutation this pins -- leaves the
    two ECEs computed on different resamples, so the delta interval opens up on nothing but
    resampling noise, and every advancement decision inherits a badly conservative interval.
    Mirrors the same property in `sibling_gap_compare`, where a pure rescale's R^2 delta is
    pinned to zero to 1e-9 for the same reason.
    """
    got = _run_main(monkeypatch, tmp_path,
                    {"ctl": CTL_PREDS, "twin": list(CTL_PREDS), "good": GOOD_PREDS})
    assert got.status == 0, got.message
    assert got.json["paired_delta_ece_ci95_vs_ref"]["twin"] == [0.0, 0.0]
    # And the bootstrap really did vary: the twin's own MARGINAL interval is wide, which is
    # exactly the conservative interval the pairing removes from the delta.
    lo, hi = got.json["ece_ci95"]["twin"]
    assert hi - lo > 0.01
    assert lo < got.json["heads"]["twin"]["ece"] < hi
    # A head that is genuinely different does NOT get a degenerate delta.
    glo, ghi = got.json["paired_delta_ece_ci95_vs_ref"]["good"]
    assert ghi > glo


def test_main_bootstraps_every_seed_it_was_given(monkeypatch, tmp_path):
    """`--boot-seeds` is a list, and taking only the first seed left the suite green.

    The reps count is what the reported interval's resolution rests on, so it is asserted as
    reps-per-seed times seeds rather than "more than zero".
    """
    got = _run_main(monkeypatch, tmp_path, {"ctl": CTL_PREDS, "good": GOOD_PREDS},
                    boot=40, boot_seeds="7,8,9,10")
    assert got.status == 0, got.message
    assert got.json["bootstrap"]["seeds"] == [7, 8, 9, 10]
    assert got.json["bootstrap"]["reps_per_seed"] == 40
    assert got.json["bootstrap"]["reps_total"] == 160


def test_main_refuses_when_the_recomputed_ece_disagrees_with_the_library(monkeypatch,
                                                                        tmp_path):
    """The file's ENTIRE stated guarantee, and deleting it left the suite green.

    The docstring's claim is that the ECE bootstrapped here IS the one in the published
    artifacts. The only thing enforcing that at runtime is this comparison, and it is
    UNCONDITIONAL -- no `--expect-ece` is passed here, deliberately, because the review of the
    help text turned on exactly that distinction.
    """
    ref_ece = vhe.ece_from(CTL_PREDS, RETS, BINS)["ece"]
    got = _run_main(monkeypatch, tmp_path, {"ctl": CTL_PREDS, "good": GOOD_PREDS},
                    library_ece={"ctl": ref_ece + 1e-8})
    assert got.status == 1
    assert "recomputed ECE" in got.message and "library" in got.message
    assert "resampling a metric that is not the one in the artifacts" in got.message
    # The tolerance, not merely the direction: 1e-8 is above LIBRARY_CROSSCHECK_TOL and 1e-12
    # is below it, so a check widened to `--expect-ece`'s 1e-6 would miss the disagreement
    # this refusal exists for.
    assert 1e-12 < vhe.LIBRARY_CROSSCHECK_TOL < 1e-8
    ok = _run_main(monkeypatch, tmp_path / "within", {"ctl": CTL_PREDS, "good": GOOD_PREDS},
                   library_ece={"ctl": ref_ece + 1e-12}, boot=20)
    assert ok.status == 0, ok.message


def test_main_refuses_a_head_whose_returns_column_is_not_the_first_heads(monkeypatch,
                                                                        tmp_path):
    """The refusal that makes the bootstrap PAIRED rather than merely simultaneous.

    Every head is meant to be scored on the same examples in the same order; if one is not,
    the shared resampled index set indexes two different datasets and every paired delta is
    meaningless. The perturbed column here is the SAME LENGTH as the reference's -- a length
    change would break other things -- so only a value comparison catches it.
    """
    other = list(RETS)
    other[17] = 1.0 if other[17] != 1.0 else -1.0
    assert len(other) == len(RETS) and other != list(RETS)
    got = _run_main(monkeypatch, tmp_path,
                    {"ctl": CTL_PREDS, "shifted": GOOD_PREDS},
                    per_head_rets={"shifted": other})
    assert got.status == 1
    assert "returns column differs from the first head's" in got.message
    assert "no paired delta below is meaningful" in got.message


def test_main_refuses_an_expect_ece_that_the_recomputation_does_not_match(monkeypatch,
                                                                         tmp_path, capsys):
    """`--expect-ece` pins the DATA and the CHECKPOINT to the published cell's, and deleting
    the comparison left the flag as documentation.

    Both sides are pinned: a figure off by more than `PUBLISHED_ECE_TOL` refuses, and the
    right figure passes and is reported as matching -- so the refusal cannot be satisfied by
    rejecting every `--expect-ece`.
    """
    ref_ece = vhe.ece_from(CTL_PREDS, RETS, BINS)["ece"]
    bad = _run_main(monkeypatch, tmp_path, {"ctl": CTL_PREDS},
                    argv_extra=("--expect-ece", f"ctl={ref_ece + 1e-5!r}"), boot=20)
    assert bad.status == 1
    assert "!= expected" in bad.message
    assert "not the one the published cell" in bad.message

    good = _run_main(monkeypatch, tmp_path / "matching", {"ctl": CTL_PREDS},
                     argv_extra=("--expect-ece", f"ctl={ref_ece!r}"), boot=20)
    assert good.status == 0, good.message
    assert "[matches published]" in capsys.readouterr().out


def test_main_records_immutable_checkpoint_and_calibration_input_identities(monkeypatch, tmp_path):
    got = _run_main(monkeypatch, tmp_path, {"ctl": CTL_PREDS, "good": GOOD_PREDS}, boot=20)
    assert got.status == 0, got.message
    assert got.json["heads"]["ctl"]["checkpoint_sha256"] == hashlib.sha256(
        (tmp_path / "ctl.pt").read_bytes()).hexdigest()
    assert got.json["heads"]["good"]["checkpoint_sha256"] == hashlib.sha256(
        (tmp_path / "good.pt").read_bytes()).hexdigest()
    assert got.json["data_inputs"] == [{
        "path": str(tmp_path / "calibration-shard"),
        "sha256": vhe.sha256_path(tmp_path / "calibration-shard"),
    }]


def test_main_refuses_unknown_or_duplicate_published_ece_expectations(monkeypatch, tmp_path):
    unknown = _run_main(monkeypatch, tmp_path, {"ctl": CTL_PREDS},
                        argv_extra=("--expect-ece", "stale=0.1"), boot=20)
    assert unknown.status == 1
    assert "not supplied --head names" in unknown.message
    duplicate = _run_main(monkeypatch, tmp_path / "duplicate", {"ctl": CTL_PREDS},
                          argv_extra=("--expect-ece", "ctl=0.1", "--expect-ece", "ctl=0.2"),
                          boot=20)
    assert duplicate.status == 1
    assert "one unique NAME=VALUE" in duplicate.message


def test_main_refuses_a_ref_that_is_not_one_of_the_heads(monkeypatch, tmp_path):
    """Every delta is taken against `--ref`, so a typo'd name must stop the run rather than
    KeyError somewhere inside the bootstrap."""
    got = _run_main(monkeypatch, tmp_path, {"ctl": CTL_PREDS}, ref="baseline", boot=20)
    assert got.status == 1
    assert "is not one of" in got.message
