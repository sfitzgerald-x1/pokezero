# Component CI tiers

The component runner partitions the original discovery blocks into disjoint light, extended and heavy tiers. Their union is `--tier all`; test bodies, inputs, assertions and custom suite order are preserved. Import failures and import-time skips stay in the light tier so they cannot disappear behind a manifest assignment. Empty shards and manifest overlap or unknown names fail closed.

- Light: 16 shards per PR, with the unchanged memory observer and 75% runner-memory bar (11.71 GiB on the standard hosted runner).
- Extended: four hosted shards on every push to main and on demand, under the same observer and compiler settings.
- Heavy: dedicated high-memory CPU infrastructure on demand and before a main-bound merge. A listed historical cap or compiler crash is incomplete test evidence, never a passing result.

## On-demand extended run

After merge, dispatch `gpu-engine-components-extended.yml` at the desired ref. Before this new workflow exists on the default branch, dispatch the existing `gpu-engine-components.yml` with input `tier=extended`; its `extended-on-demand` job calls the same extended workflow at the chosen ref. This gives the stacked PR a real on-demand run without changing the default branch.

```sh
gh workflow run gpu-engine-components.yml --ref BRANCH -f tier=extended
```

## Heavy route

Launch a dedicated CPU-only pod with the private `deployment/v2/launch.py` (`--gpus 0`), using a neutral run name and a node already carrying workloads. Keep infrastructure identifiers and private runtime locations in private launch configuration. Use the CI Python/JAX/NumPy/Torch/Node versions, pinned source oracle and 1,024-team fixture from the workflow. After sourcing any environment file, re-export:

```sh
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export XLA_FLAGS='--xla_cpu_parallel_codegen_split_count=1 --xla_backend_optimization_level=3'
```

Read the pod's actual `memory.max` and `cpu.max`, then choose concurrency from those limits. Use four CPU cores per execution lane. Very large modules must run alone with enough memory for the manifest's measured peak; a capped peak is only a lower bound. Retain terminal receipts, memory reports, full logs and failing test IDs. Release the pod after the run.

For each `index` from 0 to 15, run the light workflow's command with only `--tier heavy` changed (same count, case-isolation and compiler options). Set `ci_memory_dir` and `CI_SUMMARY` to fresh result locations:

```sh
python scripts/run_component_memory_check.py --output "$ci_memory_dir" \
  --label "shard-$index" --summary "$CI_SUMMARY" -- \
  python scripts/run_gpu_component_shard.py --tier heavy --index "$index" --count 16 --process-blocks \
  --isolate-standard-class test_gpu_engine_execution_runtime_plumbing.ExecutionRuntimePlumbingTests \
  --isolate-standard-class test_gpu_engine_status_kernel_owner_drain.StatusKernelOwnerDrainTests \
  --low-optimization-case test_gpu_engine_color_change_owning.ColorChangeOwningTests.test_actual_original_frame_both_views_public_types_requests_and_physics \
  --low-optimization-case test_gpu_engine_bulk_up_owning.BulkUpOwningTests.test_actual_original_frame_both_owners_witness_encoder_and_physics \
  --low-optimization-case test_gpu_engine_belly_drum_owning.BellyDrumOwningTests.test_actual_original_frame_both_owners_witness_encoder_and_physics \
  --low-optimization-case test_gpu_engine_primary_plain_status_owning.PrimaryPlainStatusTests.test_actual_original_frame_both_owners_witness_encoder_and_physics \
  --low-optimization-case test_gpu_engine_status_kernel_owner_drain.StatusKernelOwnerDrainTests.test_actual_kernel_status_spans_drain_both_original_owner_consumers \
  --low-optimization-case test_gpu_engine_secondary_status_frame_runtime.SecondaryStatusFrameRuntimeTests.test_actual_secondary_receipt_survives_end_turn_and_both_full_consumers \
  --low-optimization-case test_gpu_engine_rough_skin_owning.RoughSkinTests.test_actual_kernel_contact_write_both_full_and_legacy_physics \
  --low-optimization-case test_gpu_engine_protect_owning.ProtectOwningTests.test_actual_success_queue_and_stall_fail_block_both_full_witness_encoder_physics \
  --low-optimization-case test_gpu_engine_status_kernel_owner_drain.StatusKernelOwnerDrainTests.test_one_row_second_owner_failure_rolls_back_both_and_does_not_drain \
  --low-optimization-case test_gpu_engine_substitute_damage_owning.SubstituteDamageTests.test_actual_create_then_hit_survive_and_break_both_consumers \
  --low-optimization-case test_gpu_engine_status_residual_runtime.StatusResidualRuntimeTests.test_original_host_guards_and_default_actual_frame_remain_intact \
  --resource-interval-seconds 30 2>&1 \
  | tee /tmp/gpu-components.log "$ci_memory_dir/component.log"
```

This route is documented, not claimed to have passed in this PR. Its gate remains 75% of the effective runner limit; it is not a hosted-light qualification. No memory threshold is increased by the tier port.

## Moved modules and evidence

Source: integration snapshot `994242ce519a1646825bb1252fc438d9b0517123`, read-only. Integration-only modules are omitted from the manifests. Shared entries retain their recorded measurements. The explicit Toxic success requirement uses measured main evidence because main's test helper differs from integration's. Main-only module measurements and any resulting assignments will be added before this PR is ready for review.

| Module | Tier | Evidence |
|---|---|---|
| `test_gpu_engine_belly_drum_owning` | heavy | integration `994242ce`: unsettled min, 94.83 GiB |
| `test_gpu_engine_bulk_up_owning` | heavy | integration `994242ce`: unsettled min, 95.04 GiB |
| `test_gpu_engine_color_change_owning` | heavy | integration `994242ce`: unsettled min, 94.94 GiB |
| `test_gpu_engine_direct_hit_full_runtime` | heavy | integration `994242ce`: 24.0 min, 32.9 GiB |
| `test_gpu_engine_kernel_risk_callsites` | heavy | integration `994242ce`: 3.7 min, 11.1 GiB |
| `test_gpu_engine_primary_plain_status_owning` | heavy | integration `994242ce`: unsettled min, 170 GiB; capped lower bound, not a completed pass |
| `test_gpu_engine_protect_owning` | heavy | integration `994242ce`: unsettled min, 65.74 GiB |
| `test_gpu_engine_source_frame_witness` | heavy | integration `994242ce`: 16.5 min, 34.23 GiB |
| `test_gpu_engine_source_witness_driver` | heavy | integration `994242ce`: 16.9 min, 35.81 GiB |
| `test_gpu_engine_toxic_success_runtime` | heavy | main run [43641647612, shard 8](https://github.com/sfitzgerald-x1/pokezero/actions/runs/43641647612/job/132590303051): 41.67 GiB gate; all 164 shard tests pass; original single-test block 376 |
| `test_gpu_engine_turnloop_prefix_owning` | heavy | integration `994242ce`: 16.3 min, 34.44 GiB |
| `test_gpu_engine_v5_complete_grid` | heavy | integration `994242ce`: 56.9 min, 69.0 GiB |
| `test_gpu_engine_v5_hypothetical_batch` | heavy | integration `994242ce`: 32.1 min, 48.97 GiB |
| `test_gpu_engine_v5_hypothetical_profile` | heavy | integration `994242ce`: 15.0 min, 10.98 GiB |
| `test_gpu_engine_baton_paralysis_input` | extended | integration `994242ce`: 7.8 min, 8.36 GiB |
| `test_gpu_engine_baton_pass_transaction` | extended | integration `994242ce`: 4.0 min, 6.35 GiB |
| `test_gpu_engine_baton_substitute_input` | extended | integration `994242ce`: 6.3 min, 7.46 GiB |
| `test_gpu_engine_baton_toxic_status` | extended | integration `994242ce`: 3.9 min, 6.2 GiB |
| `test_gpu_engine_before_move_blocked_input` | extended | integration `994242ce`: 4.0 min, 6.15 GiB |
| `test_gpu_engine_earthquake_levitate_input` | extended | integration `994242ce`: 4.6 min, 6.23 GiB |
| `test_gpu_engine_encore_owning` | extended | integration `994242ce`: 3.6 min, 6.35 GiB |
| `test_gpu_engine_flash_fire_end_owning` | extended | integration `994242ce`: 6.9 min, 8.16 GiB |
| `test_gpu_engine_focus_punch_input` | extended | integration `994242ce`: 4.0 min, 6.19 GiB |
| `test_gpu_engine_heal_bell_owning` | extended | integration `994242ce`: 8.7 min, 8.08 GiB |
| `test_gpu_engine_hidden_power_substitute_input` | extended | integration `994242ce`: 3.7 min, 6.28 GiB |
| `test_gpu_engine_levitate_twisted_spoon_input` | extended | integration `994242ce`: 3.8 min, 6.08 GiB |
| `test_gpu_engine_natural_cure_departure_input` | extended | integration `994242ce`: 7.4 min, 8.27 GiB |
| `test_gpu_engine_no_damage_owning` | extended | integration `994242ce`: 6.8 min, 8.41 GiB |
| `test_gpu_engine_primary_sleep_miss` | extended | integration `994242ce`: 4.1 min, 5.97 GiB |
| `test_gpu_engine_rest_success_input` | extended | integration `994242ce`: 4.9 min, 6.85 GiB |
| `test_gpu_engine_return_substitute_input` | extended | integration `994242ce`: 3.7 min, 6.18 GiB |
| `test_gpu_engine_secondary_nonproc_runtime` | extended | integration `994242ce`: 7.5 min, 8.27 GiB |
| `test_gpu_engine_self_destruct_conditional_abilities` | extended | integration `994242ce`: 3.6 min, 6.41 GiB |
| `test_gpu_engine_self_destruct_input` | extended | integration `994242ce`: 3.9 min, 6.54 GiB |
| `test_gpu_engine_thunder_wave_dormant_water_absorb` | extended | integration `994242ce`: 3.7 min, 6.23 GiB |
| `test_gpu_engine_thunder_wave_ground_immunity` | extended | integration `994242ce`: 3.6 min, 6.36 GiB |
| `test_gpu_engine_thunder_wave_limber_input` | extended | integration `994242ce`: 4.1 min, 6.44 GiB |
| `test_gpu_engine_thunder_wave_substitute_failure` | extended | integration `994242ce`: 3.7 min, 6.21 GiB |
| `test_gpu_engine_toxic_runtime` | extended | integration `994242ce`: 6.6 min, 8.17 GiB |
| `test_gpu_engine_v5_depth1_provider` | extended | integration `994242ce`: 8.2 min, 7.91 GiB |
| `test_gpu_engine_v5_public_world` | extended | integration `994242ce`: 7.7 min, 7.59 GiB |
| `test_gpu_engine_variable_direct_runtime` | extended | integration `994242ce`: 6.4 min, 7.96 GiB |
| `test_gpu_engine_will_o_wisp_miss_input` | extended | integration `994242ce`: 4.3 min, 6.29 GiB |
| `test_gpu_engine_wonder_guard_immunity` | extended | integration `994242ce`: 8.3 min, 8.15 GiB |
