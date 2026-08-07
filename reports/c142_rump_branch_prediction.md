# C142 — rump-branch adjudication: prediction, registered before the sweeps

Committed **before** any sweep runs. The change under test is a harness change that can only
ever *remove* reported divergences, so a prediction written afterwards would be worthless.

## The change

`scripts/engine_transition_differential.py`, `evaluate_boundary_strict`. The matcher's
contract is existential — *some* enumerated branch reproduces the observation.
`strict:lossy_render` drops individual branches before comparison; when the branch that
would have matched is one of the dropped ones, the surviving **rump** set makes the
existential unverifiable rather than false, and adjudicating anyway reports a matched
boundary as divergent.

After the change, a boundary where (a) nothing that survived the filter reproduced the
observation and (b) at least one positive-mass branch was dropped returns a new verdict
`skip_rump`, counted as `skip:rump_branch_set` and **never** folded into
`transition:diverged`. The surviving mass fraction is recorded
(`skip:rump_branch_set_surviving_decile:N`) and each drop is attributed to its marker
(`strict:lossy_render_marker:*`), neither of which existed before.

## What is already established, by replay rather than by sweep

Row `19200131/129` of `reports/artifacts/c141_final_holdout_sweep.json`, replayed from its
retained state (no re-measurement of that window):

| | mass | render | recoil component |
|---|---|---|---|
| non-crit arm | 93.75 % | **dropped** — `attract_empty_tail_ambiguous:paralyzed+cannot_act` | `('recoil', -19)` |
| crit arm | 6.25 % | usable | `('recoil', -32)` (capped-lethal) |
| observation | — | — | `('recoil', -18)` |

The verdict rested on **6.25 %** of the enumerated mass. Allowlisting only that one marker
turns the same boundary into `matched`. So the reported `roll_scaled_component` divergence
is a rump-branch artifact and there is no recoil defect.

## Prediction

**`skip:rump_branch_set` = 0 on dev `19,000,000–199` and 0 on the validation holdout
`19,100,000–199`, and `transition:matched` / `transition:diverged` identical between the
baseline and the fixed run on both windows.**

Reasoning, stated so it can be checked rather than reinterpreted:

- Dev has never recorded a single `strict:lossy_render` on any artifact in
  `reports/artifacts/` (the counter is absent from all 27 dev sweeps), so no dev boundary
  has a branch to drop and the new exit is unreachable there.
- The validation holdout records `strict:lossy_render` = 3 on every artifact from `c121`
  through `c138`, alongside `transition:diverged` = 0 on the current build. A boundary that
  matches on its rump set never reaches the new exit, so those 3 drops should produce 0
  withheld verdicts.
- `strict:lossy_render` was **14** on the final holdout against 3 and 0 on the other two
  windows. That is the population this change addresses and it is largest exactly where it
  cannot be re-measured.

## This measurement cannot confirm the change

Stated up front so a null result is not later read as support. If the prediction holds, both
permitted windows are **silent** on the new exit, and the only evidence that the change does
the right thing is the retained-state replay above plus
`tests/test_rump_branch_adjudication.py`. The sweeps establish something narrower and still
worth having: that the change is **behaviour-preserving on both development windows** — it
does not withhold verdicts that the current harness resolves.

## Falsifier

**"Nothing opened, nothing closed."**

- If `transition:diverged` **falls** on dev or the validation holdout between the baseline
  and the fixed run, the change removes reported divergences on a window that has been swept
  many times, its blast radius is larger than claimed, and every removed row must be named
  and replayed before the change is defensible.
- If `transition:diverged` **rises** on either window, the two runs differ by more than the
  patch and the comparison is not single-variable — the result is void, not favourable.
- If `skip:rump_branch_set` **> 0** on either window, the population is live outside the
  final holdout and each withheld row must be replayed and reported individually rather than
  summarised as a count.
- If `transition:diverged != strict:diverged_on_full_branch_set` in the fixed run, the
  invariant the change is supposed to establish — every reported divergence rests on 100 %
  of its enumerated mass — does not hold, and the implementation is wrong.

Both windows are run at 200 games, `--matcher strict`, roll enumeration off (the shipping
path), on one build, with the baseline taken from a `git worktree` at the branch's merge base
`cc6ce904` so the two sides differ only by the patch. **No run at or above seed 19,200,000.**
