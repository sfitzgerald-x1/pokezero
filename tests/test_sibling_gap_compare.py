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
