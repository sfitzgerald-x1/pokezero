# k0 depth grid — artifacts (2026-07-29)

Head-to-head cells: engine MCTS vs the SAME checkpoint's raw policy, seeds
600000+, seats mirrored by seed parity, s1024 / batch 64 / worlds 4.

- `results/k0g-*.json` — k0 checkpoint `v3hist-k0-enthalf-5m-20260724` @ `iteration-2519`
  (`transition_token_budget = 0`: Markov, no history region content).
- `results/k64f-*.json` — k64 checkpoint `v3hist-k64-enthalf-5m-20260723` @ `iteration-2657`
  (`transition_token_budget = 64`), re-run against FRESHLY exported encoder
  tables. The §9 grid reused a cached tables file carrying a 1233-token vocab.

Arm codes in the `arm` field:

| arm | meaning |
|---|---|
| `ctl` | raw vs raw, same checkpoint — measures the null |
| `c` | tables' `default_feature_masks` derived from the checkpoint |
| `a` | tables' masks left as the shipped exporter emitted them |
| `k64c` | k64, fresh vocab + checkpoint-derived masks |

`a` and `c` are provably the same experiment: `engine_search.
_latch_encoder_tables_to_model_config` overwrites the tables' masks with the
checkpoint's before the crate parses them. The `a-d1` vs `c-d1` pair is retained
as the empirical proof of that (bit-identical on all 100 seeds).

Every cell records its own provenance under `provenance`, including the tables
it ran against and any `mask_drift` between those tables and the checkpoint.

Regenerate the tables with:

    PYTHONPATH=src python scripts/k0_grid_report.py --results <this dir>/results
