# Incumbent HP evaluator comparison

Phase A now has an explicit HP valuation option on the incumbent's existing encoded model tree. It retains the frozen champion's policy priors and model forwards, the world sampler, the depth 6 and batch 16 driver, and the four-world allocation. This is an evaluator comparison, not a replacement with the separate HP search tree or the uniform-action rollout mode. It has not established a playing-strength gain or admitted the prospective panels.

## Evaluator and evidence contract

`EngineMctsConfig.model_leaf_override="hp_fraction"` changes nonterminal leaf valuation. Terminal branches retain their existing terminal outcomes. The HP evaluator produces a side-one-absolute value, `0.5 + 0.5 * (side_one_hp_fraction - side_two_hp_fraction)`, so it must not receive the seat reflection used by the checkpoint's self-relative value head.

The option requires the strict model-prior path and fixed allocation; it rejects simultaneous rollout, shadow-rollout, early-stopping and own-policy-opponent modes. With the option absent, historical native call shapes remain unchanged. HP keeps the model encodings and forwards even when a leaf needs only valuation, preserving the original encoding/refusal boundary and policy-prior computation.

Each native invocation reports its HP rows, model evaluations, completed iterations, value frame and retained-forward mode. Decision evidence retains those invocations without multiplying compute by duplicate belief weights. The adapter reconciles completed work to the existing telemetry, rejects absent or unrequested HP evidence, and does not claim that HP work was a raw-policy terminal rollout.

## Verification

Five compiled tests exercise actual HP valuation, champion root-prior retention on the synthetic model fixture, both seat frames, inert rollout knobs, and the existing timed batched driver. An additional eight contract tests cover positional compatibility, unsupported configurations, missing and malformed evidence, and compute conservation. Final regressions passed with the isolated model wheel: 388 arbiter, conversion, calibration and mutation-evidence tests; 127 prospective harness tests; and 342 engine/privacy tests, including all 83 privacy-contract tests. No tests in these groups were skipped.

The full frozen champion also matched its unchanged TorchScript export and native evaluation exactly on synthetic encoded batches of 1, 8 and 16. The observed maximum absolute differences were zero for the eager/export comparison, policy logits, values and policy priors. This checks inference parity, not battle fidelity or action quality.

The first new mutation sweep was interrupted with exit 130 because its runner discarded the isolated native package path and imported an older installed wheel. Five unrelated HP failures contaminated its mutant verdicts. None is accepted as mutation evidence. All eight target and killer files were restored byte-for-byte. The runner now keeps the current source first while preserving the isolated native dependency path; its unmodified control passed all 151 tests. The separate fresh sweep completed with exit 0: all 66 mutations were caught, all seven classifier controls produced their required verdicts, and source restoration was verified. Its current-source evidence is `reports/artifacts/rollout_leaf_witness_mutation_battery_20261010_hp.json`. The original historical battery remains unchanged and has a separate byte-preservation test.

## Excluded runtime qualification

`scripts/qualify_search_over_raw_source.py --incumbent-hp` defines a separate ten-second engineering qualification: one excluded raw source game, two sampled non-opening requests, and eight terminal continuations per unique selected action. Its selectors are raw, incumbent model, reference model, incumbent HP, incumbent team-only oracle model, and incumbent team-only oracle HP. Each nonraw adapter is closed after one selection.

Registration must bind clean source, simulator, champion, encoder tables, native build and reference factory settings before launch. Any failed attempt remains failed and is not retried. Oracle inputs contain only the original opponent team, never a committed opponent action or the current private source snapshot. These roots cannot join exploration or validation. A forced-action request checks plumbing but cannot demonstrate better action selection.

The runtime qualification remains pending. Even a successful qualification would be technical evidence only; broader midgame coverage and measured resource ceilings are still required.

## Remaining scientific gates

The incumbent raw-policy terminal evaluator remains unimplemented. Phase A still needs the complete historical exposure inventory, actual collection workers bound to the fixed fresh roster, direct belief/value/fidelity diagnostics, exploration selection and sealed validation. Phase B remains conditional on a deployable gain, qualified Foul Play integration and valid joint inference with missing outcomes and stopping. Any training extension needs fresh validation seeds. The stopped historical 64-cluster, 256-game experiment and its 90 accepted outcomes are not resumed or pooled into prospective evidence.
