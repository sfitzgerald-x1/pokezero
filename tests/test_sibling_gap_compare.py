"""Known-answer tests for `scripts/sibling_gap_compare.py`.

This comparator is what decides whether a Phase 3 arm advanced, so it is tested the way a
metric has to be tested: by feeding it data whose answer is already known and asserting it
recovers that answer. Twice in this programme a headline metric could not tell a perfect
subject from a hopeless one, and both times the fault was visible the moment a synthetic case
with a known answer was pushed through.

The four synthetic cells are the four outcomes an advancement decision has to distinguish:

  ref      an attenuated but informative view of the latent sibling gap
  rescale  ref multiplied by a constant -- NO new information at all
  better   the same slope with less idiosyncratic error
  worse    the same slope with more

`rescale` is the important one. It is exactly what a margin-0 pairwise ranking loss pushes a
head toward, and a comparator that cannot separate it from `better` would certify a training
programme off arithmetic on the output spread.
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "sibling_gap_compare", REPO / "scripts" / "sibling_gap_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sgc = _load()
NOISE_VAR = 0.004261        # the banked mean, so the attenuation factor is the real one


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """465 pairs, matching the real bank's size and label noise."""
    d = tmp_path_factory.mktemp("synth")
    rng = random.Random(11)
    n = 465
    latent = [rng.gauss(0, 0.10) for _ in range(n)]
    truth = [x + rng.gauss(0, NOISE_VAR ** 0.5) for x in latent]
    ref = [0.30 * x + rng.gauss(0, 0.02) for x in latent]
    cells = {
        "ref": ref,
        "rescale": [3.3333 * h for h in ref],
        "better": [0.30 * x + rng.gauss(0, 0.004) for x in latent],
        "worse": [0.30 * x + rng.gauss(0, 0.06) for x in latent],
    }
    for name, hg in cells.items():
        (d / f"{name}.json").write_text(json.dumps({"pairs": [
            {"seed": i, "prefix": 0, "seat": "p1", "head_gap": h, "true_gap": t,
             "noise_var": NOISE_VAR}
            for i, (h, t) in enumerate(zip(hg, truth))]}))
    return d


def _run(synth, ref="ref", boot=600):
    import subprocess
    import sys
    cmd = [sys.executable, str(REPO / "scripts" / "sibling_gap_compare.py"),
           "--ref", ref, "--boot", str(boot), "--json", str(synth / "out.json")]
    for name in ("ref", "rescale", "better", "worse"):
        cmd += ["--cell", f"{name}={synth / (name + '.json')}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads((synth / "out.json").read_text())


def test_pure_rescale_moves_beta_and_leaves_r2_exactly_untouched(synth):
    out = _run(synth)
    d = out["paired_delta_ci_vs_ref"]["rescale"]
    lo, hi = d["d_beta_corrected"]
    assert lo > 0, "a x3.33 rescale must register as a beta move"
    # R^2 is scale-invariant, so the delta and BOTH ends of its interval are zero to
    # floating point. This is the assertion that makes the comparator able to call (b).
    assert out["point"]["rescale"]["r2"] == pytest.approx(out["point"]["ref"]["r2"], abs=1e-12)
    assert d["d_r2"][0] == pytest.approx(0.0, abs=1e-9)
    assert d["d_r2"][1] == pytest.approx(0.0, abs=1e-9)


def test_pure_rescale_is_identified_as_a_rescale_by_the_mechanism_test(synth):
    out = _run(synth)
    p = out["point"]["rescale"]
    assert p["corr_head_gap_with_ref"] == pytest.approx(1.0, abs=1e-9)
    assert p["sd_ratio_vs_ref"] == pytest.approx(3.3333, rel=1e-3)
    # The measured beta lands exactly on what the spread ratio alone predicts: the move is
    # arithmetic, not information.
    assert p["beta_head_on_true_noise_corrected"] == pytest.approx(
        p["implied_beta_corrected_from_pure_rescale"], rel=1e-9)


def test_a_genuinely_better_head_moves_r2_and_not_beta(synth):
    out = _run(synth)
    d = out["paired_delta_ci_vs_ref"]["better"]
    assert d["d_r2"][0] > 0, "less idiosyncratic error must raise R^2 beyond noise"
    blo, bhi = d["d_beta_corrected"]
    assert blo < 0 < bhi, "and it need not move beta at all -- which is why R^2 is the gate"
    assert out["point"]["better"]["corr_head_gap_with_ref"] < 0.99


def test_a_worse_head_moves_r2_down(synth):
    out = _run(synth)
    assert out["paired_delta_ci_vs_ref"]["worse"]["d_r2"][1] < 0


def test_beta_alone_cannot_tell_better_from_worse(synth):
    """The reason the beta component of the gate is not usable as written.

    `better` and `worse` differ by 0.36 of R^2 -- most of the explainable variance -- and both
    of their beta deltas straddle zero. A gate reading beta would score them the same.
    """
    out = _run(synth)
    for cell in ("better", "worse"):
        lo, hi = out["paired_delta_ci_vs_ref"][cell]["d_beta_corrected"]
        assert lo < 0 < hi, f"{cell}: beta delta unexpectedly resolved"
    assert (out["point"]["better"]["r2"] - out["point"]["worse"]["r2"]) > 0.3


def test_attenuation_factor_is_shared_and_equals_the_r2_ceiling(synth):
    """Every cell reuses one ground-truth column, so the correction is a shared constant.

    That has a consequence worth pinning: the noise correction cannot change the ORDERING or
    the RATIO of two cells' betas, only their absolute level against the handoff. A reader
    comparing cells may use either axis; a reader comparing to the 3.28x baseline may not.
    """
    out = _run(synth)
    a = out["attenuation_factor_shared"]
    assert out["r2_ceiling"] == pytest.approx(a)
    for cell in ("ref", "rescale", "better", "worse"):
        p = out["point"][cell]
        assert p["beta_head_on_true_noise_corrected"] == pytest.approx(
            p["beta_head_on_true_raw"] / a, rel=1e-12)


def test_it_refuses_cells_that_disagree_on_the_reused_ground_truth(synth, tmp_path):
    """Two cells scored against different true_gap columns are two experiments."""
    import subprocess
    import sys
    bad = tmp_path / "bad.json"
    rows = json.loads((synth / "ref.json").read_text())["pairs"]
    for r in rows:
        r["true_gap"] = r["true_gap"] + 0.01
    bad.write_text(json.dumps({"pairs": rows}))
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "sibling_gap_compare.py"),
         "--ref", "ref", "--boot", "10",
         "--cell", f"ref={synth / 'ref.json'}", "--cell", f"bad={bad}"],
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert "true_gap column differs" in (proc.stdout + proc.stderr)


# ---------------------------------------------------------------------------------------
# Known answers for the quantities the four synthetic cells above pin only up to a
# transform, plus the two refusals the tool needs and did not have.
# ---------------------------------------------------------------------------------------

# Every number in the fixtures below is a dyadic rational, so the arithmetic is EXACT in
# binary floating point and the expected attenuation factor is a known answer rather than the
# formula restated back at itself:
#     var(true_gap) = 0.015625, mean(noise_var) = 0.00390625, 1 - 0.25 = 0.75
DYADIC_TRUTHS = tuple(0.125 if i % 2 else -0.125 for i in range(24))
DYADIC_TRUTH_VAR = 0.015625
DYADIC_NOISE_VAR = 0.00390625


def _write_cell(path, head_gaps, truths, noise_var):
    path.write_text(json.dumps({"pairs": [
        {"seed": i, "prefix": 0, "seat": "p1", "head_gap": h, "true_gap": t,
         "noise_var": noise_var}
        for i, (h, t) in enumerate(zip(head_gaps, truths))]}))
    return path


def _run_cli(tmp_path, cells, ref, *, boot=50, json_name="cli-out.json"):
    """Run the comparator with cells in the given COMMAND-LINE ORDER."""
    import subprocess
    import sys
    out = tmp_path / json_name
    cmd = [sys.executable, str(REPO / "scripts" / "sibling_gap_compare.py"),
           "--ref", ref, "--boot", str(boot), "--json", str(out)]
    for name, path in cells:
        cmd += ["--cell", f"{name}={path}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc, out


def _dyadic_pair_of_cells(tmp_path, *, noise_var=DYADIC_NOISE_VAR):
    truths = list(DYADIC_TRUTHS)
    ref = [0.5 * t + 0.015625 * ((i % 5) - 2) for i, t in enumerate(truths)]
    other = [0.25 * t + 0.03125 * ((i % 7) - 3) for i, t in enumerate(truths)]
    return [("ref", _write_cell(tmp_path / "d-ref.json", ref, truths, noise_var)),
            ("other", _write_cell(tmp_path / "d-other.json", other, truths, noise_var))]


def test_the_attenuation_factor_is_a_known_answer_not_a_self_consistent_ratio(tmp_path):
    """The formula itself, pinned against a hand-built case.

    The existing attenuation test asserts only that `beta_corrected == beta_raw / atten`, which
    is true of ANY atten -- `atten = 1.0` (no correction at all) and the sign-flipped
    `1.0 + noise/var` both satisfy it. The factor scales every beta this tool reports and the
    3.28x handoff baseline is read off it, so it needs a number, not a relationship.
    """
    cells = _dyadic_pair_of_cells(tmp_path)
    proc, out = _run_cli(tmp_path, cells, "ref")
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    assert doc["label_variance"] == pytest.approx(DYADIC_NOISE_VAR, abs=1e-15)
    assert doc["attenuation_factor_shared"] == pytest.approx(0.75, abs=1e-15)
    assert doc["r2_ceiling"] == pytest.approx(0.75, abs=1e-15)
    assert "attenuation factor 0.750000" in proc.stdout
    # 1 - noise/var, not 1 + it. The correction must INFLATE the raw slope -- label noise on
    # the regressor attenuates it -- so a factor above 1.0 moves every beta the wrong way, and
    # here it would understate the corrected slope by 40% while still passing the existing
    # self-consistency test.
    assert doc["attenuation_factor_shared"] < 1.0
    assert doc["attenuation_factor_shared"] == pytest.approx(
        1.0 - DYADIC_NOISE_VAR / DYADIC_TRUTH_VAR, abs=1e-15)
    for cell in ("ref", "other"):
        p = doc["point"][cell]
        assert abs(p["beta_head_on_true_noise_corrected"]) > abs(p["beta_head_on_true_raw"])


def test_it_refuses_when_the_banked_label_noise_exceeds_the_observed_gap_variance(tmp_path):
    """A non-positive attenuation factor must stop the run, not scale it.

    With `noise_var` above the observed variance of `true_gap` the tool printed attenuation
    -9.94, divided every beta by it -- SIGN-FLIPPING all of them -- printed a confident MOVED
    off an interval that was the negative of the one it meant, and exited 0. There is no
    reading of that output worth having, and nothing in it says so.
    """
    # Only `noise_var` moves; the truth column and both head_gap columns are the fixture above.
    cells = _dyadic_pair_of_cells(tmp_path, noise_var=11.0 * DYADIC_TRUTH_VAR)
    proc, _ = _run_cli(tmp_path, cells, "ref", boot=10)
    both = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "REFUSING" in both
    assert "attenuation factor -10.000000 is not positive" in both
    assert "MOVED" not in both
    assert "PAIRED DELTAS" not in both


def test_it_refuses_a_cell_whose_head_gap_column_has_no_variance(tmp_path):
    """A COLLAPSED value head -- constant output, hence constant head_gap -- is a live outcome
    of the training this comparator scores. `ols` signals it by returning None, which used to
    be unpacked into a bare TypeError naming neither the cell nor the reason.
    """
    truths = list(DYADIC_TRUTHS)
    ref = [0.5 * t + 0.015625 * ((i % 5) - 2) for i, t in enumerate(truths)]
    cells = [("ref", _write_cell(tmp_path / "c-ref.json", ref, truths, DYADIC_NOISE_VAR)),
             ("collapsed", _write_cell(tmp_path / "c-flat.json", [0.01] * len(truths), truths,
                                       DYADIC_NOISE_VAR))]
    proc, _ = _run_cli(tmp_path, cells, "ref", boot=10)
    both = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "no variance to regress" in both
    assert "'collapsed'" in both


def test_the_rescale_test_survives_the_ref_being_listed_second(synth, tmp_path):
    """`implied_beta_corrected_from_pure_rescale` reads the REF cell's beta, so computing it
    inside the per-cell loop left every cell listed BEFORE the ref with None -- which the
    RESCALE TEST formats with `:.4f`.

    The mechanism test that separates "learned something" from "rescaled its output" was
    therefore dead on argument ORDER alone: fine with the ref first, TypeError with it second.
    """
    first, out_first = _run_cli(
        tmp_path, [("ref", synth / "ref.json"), ("rescale", synth / "rescale.json")], "ref",
        json_name="ref-first.json")
    assert first.returncode == 0, first.stderr
    second, out_second = _run_cli(
        tmp_path, [("rescale", synth / "rescale.json"), ("ref", synth / "ref.json")], "ref",
        json_name="ref-second.json")
    assert second.returncode == 0, second.stderr

    assert "RESCALE TEST" in second.stdout
    assert "implied by spread alone" in second.stdout
    assert "None" not in second.stdout

    doc = json.loads(out_second.read_text())
    p = doc["point"]["rescale"]
    assert p["implied_beta_corrected_from_pure_rescale"] is not None
    assert p["implied_beta_corrected_from_pure_rescale"] == pytest.approx(
        doc["point"]["ref"]["beta_head_on_true_noise_corrected"] * p["sd_ratio_vs_ref"],
        rel=1e-12)
    # Order must not change a single reported number.
    assert doc["point"] == json.loads(out_first.read_text())["point"]


def test_ols_reports_r_squared_and_not_a_monotone_transform_of_it(synth):
    """R^2 was pinned only up to a monotone transform.

    Every existing assertion compares two cells' R^2, or a delta against zero, and
    `sqrt(beta_th * beta_ht)` satisfies all of them -- while printing |r| in a column labelled
    R^2, i.e. 0.80 of variance explained where the answer is 0.64. The two slopes are pinned
    separately here too, because they are deliberately different quantities and the tool
    reports both: `beta_head_on_true` is the gate's convention, `beta_true_on_head` is the
    calibration slope.
    """
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.0, 6.0, 4.0, 8.0]
    beta_head_on_true, beta_true_on_head, r2 = sgc.ols(xs, ys)
    assert beta_true_on_head == pytest.approx(1.6)
    assert beta_head_on_true == pytest.approx(0.4)
    assert sgc.pearson(xs, ys) == pytest.approx(0.8)
    assert r2 == pytest.approx(0.64)
    assert r2 == pytest.approx(sgc.pearson(xs, ys) ** 2)
    assert r2 != pytest.approx(0.8, abs=1e-6), "R^2, not |r| -- sqrt(beta_th*beta_ht)"
    assert sgc.ols([1.0, 1.0, 1.0], ys[:3]) is None
